import os, pickle
import tensorflow as tf
from tensorflow import keras

import config
from caption_common import BeamCaptioner, SinusoidalTransformerDecoder


class EsCaptioner(BeamCaptioner):
    def __init__(self):
        # Keras tokenizer pickled at training time (word_index / index_word).
        if not os.path.exists(config.ES_TOKENIZER_PKL):
            raise RuntimeError(f"es tokenizer not found: {config.ES_TOKENIZER_PKL}")
        with open(config.ES_TOKENIZER_PKL, "rb") as f:
            self.tokenizer = pickle.load(f)
        self.word_index = self.tokenizer.word_index
        self.index_word = self.tokenizer.index_word
        self.max_length = config.ES_MAX_LEN
        self.n_steps = config.ES_MAX_LEN - 1
        self.start_id = self.word_index["<start>"]
        self.end_id   = self.word_index["<end>"]
        self.unk_id   = self.word_index.get("<unk>")

        # Frozen visual encoder: ConvNeXt-Base without top -> 7x7x1024 -> 49x1024.
        base = keras.applications.ConvNeXtBase(include_top=False, weights="imagenet")
        self.feature_extractor = keras.Model(base.input, base.layers[-1].output)

        # Transformer decoder + load weights. pos_max_len == max_length (table sized
        # exactly on the es training generation length).
        self.model = SinusoidalTransformerDecoder(
            config.ES_VOCAB_SIZE, config.ES_MAX_LEN, config.ES_D_MODEL,
            config.ES_NUM_HEADS, config.ES_DFF, config.ES_NUM_LAYERS, config.ES_DROPOUT)
        self.model(tf.zeros((1, config.ES_FEAT_LEN, config.ES_FEAT_DIM)),
                   tf.zeros((1, 1), tf.int32), training=False)
        if not os.path.exists(config.ES_WEIGHTS):
            raise RuntimeError(f"es captioning weights not found: {config.ES_WEIGHTS}")
        self.model.load_weights(config.ES_WEIGHTS)
        self.ckpt_name = config.ES_WEIGHTS

    def _features(self, img01):
        x = tf.image.resize(tf.convert_to_tensor(img01, tf.float32),
                            (config.ES_IMG_SIZE, config.ES_IMG_SIZE)) * 255.0
        x = keras.applications.convnext.preprocess_input(x)   # ConvNeXt: identity ([0,255])
        fmap = self.feature_extractor(tf.expand_dims(x, 0))
        return tf.reshape(fmap, (1, config.ES_FEAT_LEN, config.ES_FEAT_DIM))

    def _decode_step(self, seq, features, state):
        logits = self.model(features, tf.constant([seq], tf.int32), training=False)
        return tf.nn.log_softmax(logits[0, -1]).numpy(), None
