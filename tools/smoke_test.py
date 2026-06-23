"""Smoke test of the TouNum core: loads the embedded models and runs the
classification -> denoising -> caption chain on a synthetic image.

Goal: quickly validate (no dataset, no external mount) that the embedded assets
load and that each stage produces an output. Launch:

    docker compose --profile tools run --rm trainer python tools/smoke_test.py

Output: one PASS/FAIL line per stage + non-zero exit code if a stage breaks.
"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image

# The app modules live in /app (container) or src/ (local).
HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in ("/app", os.path.join(os.path.dirname(HERE), "src")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

import config        # noqa: E402
import pipeline      # noqa: E402
import models        # noqa: E402


def _synth_image(path, size=256):
    """Write a synthetic RGB image (gradient + noise) to disk."""
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

    # Stage 1 — classification cascade
    try:
        clf = models.get_classifier()
        preds = clf.classify([img_path])
        assert preds and "class" in preds[0]
        print(f"PASS  classifier  -> class={preds[0]['class']} "
              f"conf={preds[0]['confidence']:.3f} review={preds[0]['needs_review']}")
    except Exception as e:
        failures.append("classifier"); print(f"FAIL  classifier  -> {e}")

    # Stage 2 — denoiser
    try:
        denoiser = models.get_denoiser()
        img01 = pipeline.read_rgb01(img_path)
        out = denoiser.denoise01(img01)
        assert out.shape == img01.shape and 0.0 <= float(out.min()) and float(out.max()) <= 1.0
        print(f"PASS  denoiser    -> shape={out.shape} range=[{out.min():.2f},{out.max():.2f}]")
    except Exception as e:
        failures.append("denoiser"); print(f"FAIL  denoiser    -> {e}")

    # Stage 3 — default captioner (config.CAPTIONER_KIND)
    try:
        captioner = models.get_captioner()
        caption = captioner.caption(pipeline.read_rgb01(img_path))
        assert isinstance(caption, str)
        print(f"PASS  captioner   -> kind={config.CAPTIONER_KIND} caption={caption!r}")
    except Exception as e:
        failures.append(f"captioner[{config.CAPTIONER_KIND}]"); print(f"FAIL  captioner   -> {e}")

    if failures:
        print(f"\nSMOKE TEST FAILED: {', '.join(failures)}")
        return 1
    print("\nSMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
