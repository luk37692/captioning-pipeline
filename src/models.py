import os
from tensorflow import keras
import config
import pipeline

_classifier = None
_denoiser = None
_captioners = {}

def _load_keras(path, what):
    if not os.path.exists(path):
        raise RuntimeError(f"{what} not found: {path}")
    return keras.models.load_model(path, compile=False)

def get_classifier():
    global _classifier
    if _classifier is None:
        binary = _load_keras(config.BINARY_MODEL_PATH, "Binary model")
        multi = _load_keras(config.MULTI_MODEL_PATH, "Multiclass model")
        _classifier = pipeline.CascadeClassifier(binary, multi)
    return _classifier

def get_denoiser():
    global _denoiser
    if _denoiser is None:
        model = _load_keras(config.DENOISER_PATH, "Denoiser")
        _denoiser = pipeline.Denoiser(model)
    return _denoiser

def _build_captioner(kind):
    if kind == "v2plus":
        import captioning_v2plus
        return captioning_v2plus.V2PlusCaptioner()
    raise RuntimeError(f"Unknown captioner: {kind!r} (expected: {', '.join(config.CAPTIONERS)})")

def get_captioner(kind=None):
    kind = kind or config.CAPTIONER_KIND
    if kind not in config.CAPTIONERS:
        raise RuntimeError(f"Unknown captioner: {kind!r} (expected: {', '.join(config.CAPTIONERS)})")
    if kind not in _captioners:
        _captioners[kind] = _build_captioner(kind)
    return _captioners[kind]
