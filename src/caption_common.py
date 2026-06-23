"""Shared building blocks for the 4 captioners (JSON tokenizer, sinusoidal
Transformer decoder, beam search). Each concrete captioner subclasses
`BeamCaptioner` and only provides its specifics (feature extraction + one decode step)."""
import os, json
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config

_SPECIAL = ("<start>", "<end>", "<pad>")


def load_json_tokenizer(json_path):
    """Load the embedded JSON tokenizer -> (word_index, index_word, vocab_size, max_length).
    Raises if the file is missing (no rebuild: inference depends on no dataset)."""
    if not os.path.exists(json_path):
        raise RuntimeError(f"Tokenizer not found: {json_path}")
    with open(json_path) as f:
        d = json.load(f)
    word_index = d["word_index"]
    index_word = {i: w for w, i in word_index.items()}
    return word_index, index_word, len(word_index), int(d["max_length"])


def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos, tf.float32)


class SinusoidalDecoderLayer(keras.layers.Layer):
    """Transformer decoder block (causal self-attn + cross-attn) shared by the
    captioners with sinusoidal positional encoding (v2+ and es)."""
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.self_attn = keras.layers.MultiHeadAttention(num_heads, d_model // num_heads)
        self.cross_attn = keras.layers.MultiHeadAttention(num_heads, d_model // num_heads)
        self.ffn = keras.Sequential([
            keras.layers.Dense(dff, activation="relu"),
            keras.layers.Dense(d_model)])
        self.ln1 = keras.layers.LayerNormalization()
        self.ln2 = keras.layers.LayerNormalization()
        self.ln3 = keras.layers.LayerNormalization()
        self.drop = keras.layers.Dropout(dropout)

    def call(self, x, image_features, training=False):
        attn = self.self_attn(query=x, value=x, key=x, use_causal_mask=True, training=training)
        x = self.ln1(x + attn)
        cattn = self.cross_attn(query=x, value=image_features, key=image_features, training=training)
        x = self.ln2(x + cattn)
        x = self.ln3(x + self.drop(self.ffn(x), training=training))
        return x


class SinusoidalTransformerDecoder(keras.Model):
    """Transformer decoder with sinusoidal positional encoding (v2+ and es).
    `pos_max_len` sizes the positional table (separate from the generation length)."""
    def __init__(self, vocab_size, pos_max_len, d_model, num_heads, dff, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.image_proj = keras.layers.Dense(d_model)
        self.token_emb = keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(pos_max_len, d_model)
        self.drop = keras.layers.Dropout(dropout)
        self.dec_layers = [SinusoidalDecoderLayer(d_model, num_heads, dff, dropout)
                           for _ in range(num_layers)]
        self.out = keras.layers.Dense(vocab_size, dtype="float32")

    def call(self, image_features, tokens, training=False):
        img = self.image_proj(image_features)
        seq_len = tf.shape(tokens)[1]
        x = self.token_emb(tokens)
        x *= tf.math.sqrt(tf.cast(self.d_model, x.dtype))
        x = x + tf.cast(self.pos_encoding[tf.newaxis, :seq_len, :], x.dtype)
        x = self.drop(x, training=training)
        for layer in self.dec_layers:
            x = layer(x, img, training=training)
        return self.out(x)


class BeamCaptioner:
    """Beam search + repeated n-gram blocking, shared by the 4 captioners.

    A weakly trained decoder tends to loop ("... person up of the ..."), hence the
    n-gram blocking. Subclasses must define, after init:
      - word_index / index_word
      - start_id / end_id (end_id = -1 if absent from vocab)
      - unk_id (None = no <unk> blocking)
      - n_steps (max number of decode steps)
    and implement `_features(img01)`, `_init_state()`, `_decode_step(seq, features, state)`.
    """
    unk_id = None

    @staticmethod
    def _banned_tokens(seq, n):
        """Tokens forbidden at the next step to avoid repeating an already-seen n-gram."""
        if n <= 0 or len(seq) < n:
            return set()
        prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
        banned = set()
        for i in range(len(seq) - n + 1):
            if tuple(seq[i:i + n - 1]) == prefix:
                banned.add(seq[i + n - 1])
        return banned

    def _init_state(self):
        return None

    def _decode_step(self, seq, features, state):
        """Returns (numpy log_probs of the next token, new state)."""
        raise NotImplementedError

    def caption(self, img01, beam_width=None, no_repeat_ngram_size=3):
        """Caption an HxWx3 float [0,1] image (beam search + n-gram blocking)."""
        beam_width = beam_width or config.MAX_CAPTION_BEAM
        features = self._features(img01)
        beams = [(0.0, [self.start_id], self._init_state(), False)]
        for _ in range(self.n_steps):
            candidates = []
            for logp, seq, state, done in beams:
                if done:
                    candidates.append((logp, seq, state, True))
                    continue
                lp, new_state = self._decode_step(seq, features, state)
                if self.unk_id is not None:
                    lp[self.unk_id] = -np.inf
                banned = self._banned_tokens(seq, no_repeat_ngram_size)
                taken = 0
                for t in np.argsort(lp)[::-1]:
                    t = int(t)
                    if t in banned and t != self.end_id:
                        continue
                    candidates.append((logp + float(lp[t]), seq + [t], new_state, t == self.end_id))
                    taken += 1
                    if taken >= beam_width:
                        break
            candidates.sort(key=lambda b: b[0] / (len(b[1]) ** 0.7), reverse=True)
            beams = candidates[:beam_width]
            if all(b[3] for b in beams):
                break
        best = max(beams, key=lambda b: b[0] / (len(b[1]) ** 0.7))
        words = [self.index_word.get(i, "<unk>") for i in best[1]]
        return " ".join(w for w in words if w not in _SPECIAL)
