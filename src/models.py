"""Chargement paresseux + cache des modèles (classifieur, débruiteur, captioners).

Les modèles Keras sont lourds (encodeurs CNN, décodeurs Transformer) : on ne les
charge qu'à la première demande, puis on les conserve en cache mémoire pour servir
les requêtes suivantes sans recharger. L'appli (`main.py`) ne connaît que les trois
accès `get_classifier` / `get_denoiser` / `get_captioner(kind)`.
"""
import os

from tensorflow import keras

import config
import pipeline

# Caches : un classifieur, un débruiteur, et un captioner PAR variante (kind).
_classifier = None
_denoiser = None
_captioners = {}


def _load_keras(path, what):
    if not os.path.exists(path):
        raise RuntimeError(f"{what} introuvable : {path}")
    # compile=False : inférence seule, évite de reconstruire l'optimiseur/les pertes.
    return keras.models.load_model(path, compile=False)


def get_classifier():
    """Cascade multiclasse + binaire (cf. config.CLASSIFIERS). Chargée une fois."""
    global _classifier
    if _classifier is None:
        binary = _load_keras(config.BINARY_MODEL_PATH, "Modèle binaire")
        multi = _load_keras(config.MULTI_MODEL_PATH, "Modèle multiclasse")
        _classifier = pipeline.CascadeClassifier(binary, multi)
    return _classifier


def get_denoiser():
    """Auto-encodeur résiduel à skip (128x128, [0,1]). Chargé une fois."""
    global _denoiser
    if _denoiser is None:
        model = _load_keras(config.DENOISER_PATH, "Débruiteur")
        _denoiser = pipeline.Denoiser(model)
    return _denoiser


def _build_captioner(kind):
    """Instancie le captioner demandé (import paresseux : TF + poids lourds)."""
    if kind == "v2plus":
        import captioning_v2plus
        return captioning_v2plus.V2PlusCaptioner()
    if kind == "es":
        import captioning_es
        return captioning_es.EsCaptioner()
    if kind == "transformer":
        import captioning_transformer
        return captioning_transformer.TransformerCaptioner()
    if kind == "gru":
        import captioning
        return captioning.Captioner()
    raise RuntimeError(f"Captioner inconnu : {kind!r} (attendu : {', '.join(config.CAPTIONERS)})")


def get_captioner(kind=None):
    """Captioner pour la variante `kind` (défaut config.CAPTIONER_KIND). Caché par kind."""
    kind = kind or config.CAPTIONER_KIND
    if kind not in config.CAPTIONERS:
        kind = config.CAPTIONER_KIND
    if kind not in _captioners:
        _captioners[kind] = _build_captioner(kind)
    return _captioners[kind]
