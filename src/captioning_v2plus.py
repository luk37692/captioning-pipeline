import os, pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from caption_common import BeamCaptioner


def _v2p_positional_encoding(max_len, d_model):
    """Interleaved sin/cos positional encoding, IDENTICAL to the training notebook
    (Livrable3_Captioning_v2plus)."""
    positions = np.arange(max_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], tf.float32)


class _CNN_Encoder(keras.Model):
    """EfficientNetB0 + reshape + Dense(embedding_dim)+relu + LayerNorm. Matches the
    embedded best_encoder.weights.h5 (groups: efficientnet / fc / dropout). A 4-D image
    input runs the CNN; the trained fc/norm project 1280 -> embedding_dim."""
    def __init__(self, embedding_dim, dropout_rate, img_size):
        super().__init__()
        self.efficientnet = keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=(img_size, img_size, 3))
        self.reshape = keras.layers.Reshape((-1, 1280))
        self.fc = keras.layers.Dense(embedding_dim)
        self.norm = keras.layers.LayerNormalization()
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        if len(x.shape) == 4:
            x = self.efficientnet(x, training=training)
            x = self.reshape(x)
        x = tf.nn.relu(self.fc(x))
        x = self.norm(x)
        return self.dropout(x, training=training)


class TransformerDecoderBlock(keras.layers.Layer):
    """Causal self-attn + cross-attn + FFN (LayerNorm eps 1e-6), matching the notebook.
    No image projection here — the encoder already projects to embedding_dim."""
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout_rate):
        super().__init__()
        self.self_attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=dropout_rate)
        self.cross_attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=dropout_rate)
        self.ff = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embedding_dim)])
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = keras.layers.Dropout(dropout_rate)
        self.drop2 = keras.layers.Dropout(dropout_rate)
        self.drop3 = keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training=False):
        attn1 = self.self_attn(query=x, key=x, value=x, use_causal_mask=True, training=training)
        x = self.norm1(x + self.drop1(attn1, training=training))
        attn2 = self.cross_attn(query=x, key=enc_output, value=enc_output, training=training)
        x = self.norm2(x + self.drop2(attn2, training=training))
        x = self.norm3(x + self.drop3(self.ff(x), training=training))
        return x


class _TransformerDecoder(keras.Model):
    """Token embedding (scaled) + interleaved positional encoding + N decoder blocks.
    Matches best_decoder.weights.h5 (groups: embedding / dec_layers / final_layer / dropout)."""
    def __init__(self, vocab_size, embedding_dim, num_heads, ff_dim, num_layers,
                 pos_max_len, dropout_rate):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = _v2p_positional_encoding(pos_max_len, embedding_dim)
        self.dec_layers = [TransformerDecoderBlock(embedding_dim, num_heads, ff_dim, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.final_layer = keras.layers.Dense(vocab_size)

    def call(self, enc_output, tokens, training=False):
        seq_len = tf.shape(tokens)[1]
        x = self.embedding(tokens)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for layer in self.dec_layers:
            x = layer(x, enc_output, training=training)
        return self.final_layer(x)


class V2PlusCaptioner(BeamCaptioner):
    """EfficientNetB0 + 4-block Transformer decoder, matching the training notebook
    (Livrable3_Captioning_v2plus) so the embedded encoder/decoder weights load."""
    def __init__(self):
        if not os.path.exists(config.V2P_TOKENIZER_PKL):
            raise RuntimeError(f"v2+ tokenizer not found: {config.V2P_TOKENIZER_PKL}")
        with open(config.V2P_TOKENIZER_PKL, "rb") as f:
            self.tokenizer = pickle.load(f)
        self.word_index = self.tokenizer.word_index
        self.index_word = self.tokenizer.index_word
        self.n_steps = config.V2P_GEN_MAX_LEN - 1
        self.start_id = self.word_index["<start>"]
        self.end_id   = self.word_index["<end>"]
        self.unk_id   = self.word_index.get("<unk>")

        # Encoder: build the full graph (4-D dummy creates the EfficientNet variables)
        # BEFORE load_weights.
        self.encoder = _CNN_Encoder(
            config.V2P_EMBEDDING_DIM, config.V2P_DROPOUT, config.V2P_IMG_SIZE)
        self.encoder(tf.zeros((1, config.V2P_IMG_SIZE, config.V2P_IMG_SIZE, 3)),
                     training=False)
        if not os.path.exists(config.V2P_ENCODER_WEIGHTS):
            raise RuntimeError(f"v2+ encoder weights not found: {config.V2P_ENCODER_WEIGHTS}")
        self.encoder.load_weights(config.V2P_ENCODER_WEIGHTS)

        # Decoder + weight restoration.
        self.model = _TransformerDecoder(
            config.V2P_VOCAB_SIZE, config.V2P_EMBEDDING_DIM, config.V2P_NUM_HEADS,
            config.V2P_FF_DIM, config.V2P_NUM_LAYERS, config.V2P_POS_MAX_LEN,
            config.V2P_DROPOUT)
        self.model(tf.zeros((1, config.V2P_FEAT_LEN, config.V2P_EMBEDDING_DIM)),
                   tf.zeros((1, 1), tf.int32), training=False)
        if not os.path.exists(config.V2P_DECODER_WEIGHTS):
            raise RuntimeError(f"v2+ decoder weights not found: {config.V2P_DECODER_WEIGHTS}")
        self.model.load_weights(config.V2P_DECODER_WEIGHTS)
        self.ckpt_name = config.V2P_DECODER_WEIGHTS

    def _features(self, img01):
        x = tf.image.resize(tf.convert_to_tensor(img01, tf.float32),
                            (config.V2P_IMG_SIZE, config.V2P_IMG_SIZE)) * 255.0
        x = keras.applications.efficientnet.preprocess_input(x)   # EfficientNet: expects [0,255]
        return self.encoder(tf.expand_dims(x, 0), training=False)  # (1, 49, embedding_dim)

    def _decode_step(self, seq, features, state):
        logits = self.model(features, tf.constant([seq], tf.int32), training=False)
        return tf.nn.log_softmax(logits[0, -1]).numpy(), None
