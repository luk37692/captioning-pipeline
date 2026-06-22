"""Captioning « es » — ConvNeXt-Base (encodeur gelé) + décodeur Transformer 6 blocs.

Modèle du Livrable 3 (notebook `Livrable_3_Captioning_es_ConvNeXtBase_Transformer`).
Contrairement au captioner `transformer`, le tokenizer EST sauvegardé (Keras Tokenizer
picklé), donc aucune reconstruction du vocabulaire. L'architecture ci-dessous est
IDENTIQUE au notebook (encodage positionnel sinusoïdal, FFN relu) — requis pour
`load_weights`.

Interface identique aux autres captioners : `.caption(img01, beam_width=None)` où
`img01` est une image HxWx3 float dans [0, 1].
"""
import os, pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Architecture (IDENTIQUE au notebook es — requis pour load_weights)
# ─────────────────────────────────────────────────────────────────────────────
def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos, tf.float32)


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.self_attn  = keras.layers.MultiHeadAttention(num_heads, d_model // num_heads)
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


class TransformerCaptioner(keras.Model):
    def __init__(self, vocab_size, max_length, d_model=512, num_heads=8,
                 dff=2048, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.image_proj = keras.layers.Dense(d_model)
        self.token_emb = keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length, d_model)
        self.drop = keras.layers.Dropout(dropout)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Captioner : ConvNeXt-Base + Transformer + poids + beam search
# ─────────────────────────────────────────────────────────────────────────────
class EsCaptioner:
    def __init__(self):
        # Tokenizer Keras picklé à l'entraînement (word_index / index_word).
        if not os.path.exists(config.ES_TOKENIZER_PKL):
            raise RuntimeError(f"Tokenizer es introuvable : {config.ES_TOKENIZER_PKL}")
        with open(config.ES_TOKENIZER_PKL, "rb") as f:
            self.tokenizer = pickle.load(f)
        self.word_index = self.tokenizer.word_index
        self.index_word = self.tokenizer.index_word
        self.max_length = config.ES_MAX_LEN
        self.start_id = self.word_index["<start>"]
        self.end_id   = self.word_index["<end>"]
        self.unk_id   = self.word_index.get("<unk>")

        # Encodeur visuel gelé : ConvNeXt-Base sans tête -> 7x7x1024 -> 49x1024.
        base = keras.applications.ConvNeXtBase(include_top=False, weights="imagenet")
        self.feature_extractor = keras.Model(base.input, base.layers[-1].output)

        # Décodeur Transformer + restauration des poids.
        self.model = TransformerCaptioner(
            config.ES_VOCAB_SIZE, self.max_length, config.ES_D_MODEL,
            config.ES_NUM_HEADS, config.ES_DFF, config.ES_NUM_LAYERS, config.ES_DROPOUT)
        # Construire les variables (1 passe à blanc) AVANT load_weights.
        self.model(tf.zeros((1, config.ES_FEAT_LEN, config.ES_FEAT_DIM)),
                   tf.zeros((1, 1), tf.int32), training=False)
        if not os.path.exists(config.ES_WEIGHTS):
            raise RuntimeError(f"Poids captioning es introuvables : {config.ES_WEIGHTS}")
        self.model.load_weights(config.ES_WEIGHTS)
        self.ckpt_name = config.ES_WEIGHTS

    def _features(self, img01):
        x = tf.image.resize(tf.convert_to_tensor(img01, tf.float32),
                            (config.ES_IMG_SIZE, config.ES_IMG_SIZE)) * 255.0
        x = keras.applications.convnext.preprocess_input(x)   # ConvNeXt: identité ([0,255])
        fmap = self.feature_extractor(tf.expand_dims(x, 0))
        return tf.reshape(fmap, (1, config.ES_FEAT_LEN, config.ES_FEAT_DIM))

    @staticmethod
    def _banned_tokens(seq, n):
        if n <= 0 or len(seq) < n:
            return set()
        prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
        banned = set()
        for i in range(len(seq) - n + 1):
            if tuple(seq[i:i + n - 1]) == prefix:
                banned.add(seq[i + n - 1])
        return banned

    def caption(self, img01, beam_width=None, no_repeat_ngram_size=3):
        """Légende une image HxWx3 float [0,1] (beam search + blocage de n-grammes)."""
        beam_width = beam_width or config.MAX_CAPTION_BEAM
        features = self._features(img01)
        beams = [(0.0, [self.start_id], False)]
        for _ in range(self.max_length - 1):
            candidates = []
            for logp, seq, done in beams:
                if done:
                    candidates.append((logp, seq, True))
                    continue
                logits = self.model(features, tf.constant([seq], tf.int32), training=False)
                lp = tf.nn.log_softmax(logits[0, -1]).numpy()
                if self.unk_id is not None:
                    lp[self.unk_id] = -np.inf
                banned = self._banned_tokens(seq, no_repeat_ngram_size)
                taken = 0
                for t in np.argsort(lp)[::-1]:
                    t = int(t)
                    if t in banned and t != self.end_id:
                        continue
                    candidates.append((logp + float(lp[t]), seq + [t], t == self.end_id))
                    taken += 1
                    if taken >= beam_width:
                        break
            candidates.sort(key=lambda b: b[0] / (len(b[1]) ** 0.7), reverse=True)
            beams = candidates[:beam_width]
            if all(b[2] for b in beams):
                break
        best = max(beams, key=lambda b: b[0] / (len(b[1]) ** 0.7))
        words = [self.index_word.get(i, "<unk>") for i in best[1]]
        return " ".join(w for w in words if w not in ("<start>", "<end>", "<pad>"))
