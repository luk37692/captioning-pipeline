import os, pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from caption_common import BeamCaptioner, SinusoidalTransformerDecoder


class V2PlusCaptioner(BeamCaptioner):
    def __init__(self):
        # Keras tokenizer pickled at training time (word_index / index_word).
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

        # FINE-TUNED EfficientNetB0 visual encoder: start from the headless arch, then
        # restore the unfrozen weights (best_encoder.weights.h5), NOT the ImageNet weights.
        base = keras.applications.EfficientNetB0(include_top=False, weights=None)
        self.feature_extractor = keras.Model(base.input, base.layers[-1].output)
        if not os.path.exists(config.V2P_ENCODER_WEIGHTS):
            raise RuntimeError(f"v2+ encoder weights not found: {config.V2P_ENCODER_WEIGHTS}")
        self.feature_extractor.load_weights(config.V2P_ENCODER_WEIGHTS)

        # Transformer decoder + weight restoration.
        self.model = SinusoidalTransformerDecoder(
            config.V2P_VOCAB_SIZE, config.V2P_POS_MAX_LEN, config.V2P_EMBEDDING_DIM,
            config.V2P_NUM_HEADS, config.V2P_FF_DIM, config.V2P_NUM_LAYERS, config.V2P_DROPOUT)
        # Build the variables (one blank pass) BEFORE load_weights.
        self.model(tf.zeros((1, config.V2P_FEAT_LEN, config.V2P_FEAT_DIM)),
                   tf.zeros((1, 1), tf.int32), training=False)
        if not os.path.exists(config.V2P_DECODER_WEIGHTS):
            raise RuntimeError(f"v2+ decoder weights not found: {config.V2P_DECODER_WEIGHTS}")
        self.model.load_weights(config.V2P_DECODER_WEIGHTS)
        self.ckpt_name = config.V2P_DECODER_WEIGHTS

    def _features(self, img01):
        x = tf.image.resize(tf.convert_to_tensor(img01, tf.float32),
                            (config.V2P_IMG_SIZE, config.V2P_IMG_SIZE)) * 255.0
        x = keras.applications.efficientnet.preprocess_input(x)   # EfficientNet: expects [0,255]
        fmap = self.feature_extractor(tf.expand_dims(x, 0))
        return tf.reshape(fmap, (1, config.V2P_FEAT_LEN, config.V2P_FEAT_DIM))

    def _decode_step(self, seq, features, state):
        logits = self.model(features, tf.constant([seq], tf.int32), training=False)
        return tf.nn.log_softmax(logits[0, -1]).numpy(), None
