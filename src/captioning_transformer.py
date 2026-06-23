import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from caption_common import BeamCaptioner, load_json_tokenizer


class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, max_len, **kw):
        super().__init__(**kw)
        self.d_model = d_model
        self.tok = layers.Embedding(vocab_size, d_model)
        self.pos = layers.Embedding(max_len, d_model)

    def call(self, x):
        T = tf.shape(x)[1]
        scale = tf.cast(tf.math.sqrt(tf.cast(self.d_model, tf.float32)), self.compute_dtype)
        return self.tok(x) * scale + self.pos(tf.range(T))


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kw):
        super().__init__(**kw)
        self.mha1 = layers.MultiHeadAttention(num_heads, d_model // num_heads, dropout=rate)
        self.mha2 = layers.MultiHeadAttention(num_heads, d_model // num_heads, dropout=rate)
        self.ffn = keras.Sequential([layers.Dense(dff, activation="gelu"), layers.Dense(d_model)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = layers.LayerNormalization(epsilon=1e-6)
        self.do = layers.Dropout(rate)

    def call(self, x, enc, pad_mask, training=False, return_attn=False):
        a1 = self.mha1(query=x, value=x, key=x, attention_mask=pad_mask,
                       use_causal_mask=True, training=training)
        x = self.ln1(x + a1)
        a2, attn = self.mha2(query=x, value=enc, key=enc,
                             return_attention_scores=True, training=training)
        x = self.ln2(x + a2)
        x = self.ln3(x + self.do(self.ffn(x), training=training))
        return x, (attn if return_attn else None)


class CaptioningTransformer(keras.Model):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dff, max_len, rate=0.1, **kw):
        super().__init__(**kw)
        self.proj = layers.Dense(d_model)
        self.posemb = PositionalEmbedding(vocab_size, d_model, max_len)
        self.dec = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.drop = layers.Dropout(rate)
        self.out = layers.Dense(vocab_size, dtype="float32")

    def call(self, inputs, training=False, return_attn=False):
        feat, dec_in = inputs
        enc = self.proj(feat)
        x = self.drop(self.posemb(dec_in), training=training)
        pad_mask = tf.cast(tf.math.not_equal(dec_in, 0), tf.bool)[:, tf.newaxis, :]
        last_attn = None
        for layer in self.dec:
            x, att = layer(x, enc, pad_mask, training=training, return_attn=return_attn)
            if return_attn:
                last_attn = att
        logits = self.out(x)
        return (logits, last_attn) if return_attn else logits


class TransformerCaptioner(BeamCaptioner):
    def __init__(self):
        self.word_index, self.index_word, self.vocab_size, self.max_length = \
            load_json_tokenizer(config.CAP_TOKENIZER_JSON)
        self.n_steps = self.max_length - 1
        self.start_id = self.word_index["<start>"]
        self.end_id = self.word_index.get("<end>", -1)

        # Frozen visual encoder: headless ConvNeXt-Tiny -> 7x7x768 map -> 49x768.
        base = keras.applications.ConvNeXtTiny(include_top=False, weights="imagenet")
        self.feature_extractor = keras.Model(base.input, base.output)

        # Transformer decoder + weight restoration (.weights.h5 from v2 training).
        self.model = CaptioningTransformer(
            self.vocab_size, config.CAP_D_MODEL, config.CAP_NUM_LAYERS,
            config.CAP_NUM_HEADS, config.CAP_DFF, self.max_length)
        # Build the variables (one blank pass) BEFORE load_weights.
        dummy_feat = tf.zeros((1, config.CAP_FEAT_LEN, config.CAP_FEAT_DIM))
        dummy_in = tf.zeros((1, 2), tf.int32)
        self.model((dummy_feat, dummy_in))
        if not os.path.exists(config.CONVNEXT_CKPT):
            raise RuntimeError(f"v2 captioning weights not found: {config.CONVNEXT_CKPT}")
        self.model.load_weights(config.CONVNEXT_CKPT)
        self.ckpt_name = config.CONVNEXT_CKPT

    def _features(self, img01):
        x = tf.image.resize(tf.convert_to_tensor(img01, tf.float32),
                            (config.CAP_IMG_SIZE, config.CAP_IMG_SIZE)) * 255.0
        x = keras.applications.convnext.preprocess_input(x)   # ConvNeXt: identity (expects [0,255])
        fmap = self.feature_extractor(tf.expand_dims(x, 0))
        return tf.reshape(fmap, (1, config.CAP_FEAT_LEN, config.CAP_FEAT_DIM))

    def _decode_step(self, seq, features, state):
        logits = self.model((features, tf.constant([seq], tf.int32)), training=False)
        return tf.nn.log_softmax(logits[0, -1]).numpy(), None
