import numpy as np
import tensorflow as tf
from tensorflow import keras

import config

AUTOTUNE = tf.data.AUTOTUNE


def _has_internal_rescaling(model):
    def scan(layers):
        for l in layers:
            if isinstance(l, (keras.layers.Rescaling, keras.layers.Normalization)):
                return True
            if hasattr(l, "layers") and scan(l.layers):
                return True
        return False
    return scan(model.layers)


def _model_size(model, fallback):
    s = model.input_shape
    if isinstance(s, list):
        s = s[0]
    h, w = s[1], s[2]
    if h is None or w is None:
        return (fallback, fallback)
    return (int(h), int(w))


def read_rgb01(path):
    """Lit une image (PNG/JPEG) en HxWx3 float [0,1] (numpy)."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return (tf.cast(img, tf.float32) / 255.0).numpy()


class CascadeClassifier:
    def __init__(self, binary_model, multi_model):
        self.bin = binary_model
        self.multi = multi_model
        self.bin_size = _model_size(binary_model, 299)
        self.multi_size = _model_size(multi_model, 299)
        self.bin_norm = not _has_internal_rescaling(binary_model)   # True => divide by 255
        self.multi_norm = not _has_internal_rescaling(multi_model)
        self.order = config.MULTI_CLASS_ORDER
        self.photo_idx = self.order.index(config.PHOTO_CLASS)
        self.paint_idx = self.order.index(config.CONFUSED_CLASS)

    def _ds(self, paths, size, normalize, batch=32):
        def load(p):
            img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
            img = tf.image.resize(img, size)
            img = tf.cast(img, tf.float32)
            return img / 255.0 if normalize else img
        return (tf.data.Dataset.from_tensor_slices(list(paths))
                .map(load, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE))

    def classify(self, paths, mc_review_thresh=None, bin_review_band=None, bin_threshold=None):
        mc_review_thresh = config.MC_REVIEW_THRESH if mc_review_thresh is None else mc_review_thresh
        bin_review_band = config.BIN_REVIEW_BAND if bin_review_band is None else bin_review_band
        bin_threshold = config.BIN_THRESHOLD if bin_threshold is None else bin_threshold
        paths = list(paths)
        if not paths:
            return []
        probs_mc = self.multi.predict(self._ds(paths, self.multi_size, self.multi_norm), verbose=0)
        scores_bin = self.bin.predict(self._ds(paths, self.bin_size, self.bin_norm), verbose=0).ravel()

        results = []
        for i, p in enumerate(paths):
            pm = probs_mc[i]
            mc_idx = int(np.argmax(pm))
            mc_conf = float(pm[mc_idx])
            bscore = float(scores_bin[i])
            routed = mc_idx in (self.photo_idx, self.paint_idx)
            final_idx = mc_idx
            if routed:
                final_idx = self.photo_idx if bscore >= bin_threshold else self.paint_idx
            is_photo = (final_idx == self.photo_idx)
            conf = (bscore if is_photo else 1.0 - bscore) if routed else mc_conf
            review = (mc_conf < mc_review_thresh) or (routed and abs(bscore - 0.5) < bin_review_band)
            results.append({
                "path": p,
                "class": self.order[final_idx],
                "mc_class": self.order[mc_idx],
                "mc_conf": mc_conf,
                "bin_score": bscore,
                "routed": routed,
                "confidence": conf,
                "is_photo": bool(is_photo),
                "needs_review": bool(review),
            })
        return results


class Denoiser:
    def __init__(self, model):
        self.model = model
        self.size = _model_size(model, config.DENOISER_SIZE)

    def denoise01(self, img01):
        x = tf.convert_to_tensor(img01, tf.float32)
        h, w = x.shape[0], x.shape[1]
        resized = tf.image.resize(x, self.size)
        out = self.model(tf.expand_dims(resized, 0), training=False)[0]
        out = tf.image.resize(out, (h, w))
        return np.clip(out.numpy(), 0.0, 1.0)
