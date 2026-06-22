"""Smoke test du cœur TouNum : charge les modèles embarqués et fait tourner la
chaîne classification -> débruitage -> légende sur une image de synthèse.

But : valider rapidement (sans COCO ni montage externe) que les assets embarqués se
chargent et que chaque étage produit une sortie. Lancement :

    docker compose --profile tools run --rm trainer python tools/smoke_test.py

Sortie : une ligne PASS/FAIL par étage + code retour non nul si un étage casse.
"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image

# Les modules applicatifs vivent dans /app (conteneur) ou src/ (local).
HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in ("/app", os.path.join(os.path.dirname(HERE), "src")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

import config        # noqa: E402
import pipeline      # noqa: E402
import models        # noqa: E402


def _synth_image(path, size=256):
    """Écrit une image RGB de synthèse (dégradé + bruit) sur disque."""
    rng = np.random.default_rng(config.SEED)
    grad = np.linspace(0, 255, size, dtype=np.float32)
    base = np.stack([np.tile(grad, (size, 1)),
                     np.tile(grad[:, None], (1, size)),
                     np.full((size, size), 128, np.float32)], axis=-1)
    noisy = np.clip(base + rng.normal(0, 25, base.shape), 0, 255).astype("uint8")
    Image.fromarray(noisy).save(path)


def main():
    failures = []
    tmp = tempfile.mkdtemp(prefix="tounum_smoke_")
    img_path = os.path.join(tmp, "synth.png")
    _synth_image(img_path)

    # Étage 1 — cascade de classification
    try:
        clf = models.get_classifier()
        preds = clf.classify([img_path])
        assert preds and "class" in preds[0]
        print(f"PASS  classifier  -> class={preds[0]['class']} "
              f"conf={preds[0]['confidence']:.3f} review={preds[0]['needs_review']}")
    except Exception as e:
        failures.append("classifier"); print(f"FAIL  classifier  -> {e}")

    # Étage 2 — débruiteur
    try:
        denoiser = models.get_denoiser()
        img01 = pipeline.read_rgb01(img_path)
        out = denoiser.denoise01(img01)
        assert out.shape == img01.shape and 0.0 <= float(out.min()) and float(out.max()) <= 1.0
        print(f"PASS  denoiser    -> shape={out.shape} range=[{out.min():.2f},{out.max():.2f}]")
    except Exception as e:
        failures.append("denoiser"); print(f"FAIL  denoiser    -> {e}")

    # Étage 3 — captioner par défaut (config.CAPTIONER_KIND)
    try:
        captioner = models.get_captioner()
        caption = captioner.caption(pipeline.read_rgb01(img_path))
        assert isinstance(caption, str)
        print(f"PASS  captioner   -> kind={config.CAPTIONER_KIND} caption={caption!r}")
    except Exception as e:
        failures.append(f"captioner[{config.CAPTIONER_KIND}]"); print(f"FAIL  captioner   -> {e}")

    if failures:
        print(f"\nSMOKE TEST FAILED : {', '.join(failures)}")
        return 1
    print("\nSMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
