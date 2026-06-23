import os
import tensorflow as tf
from tensorflow import keras

import config
from caption_common import BeamCaptioner, load_json_tokenizer


class CNN_Encoder(keras.Model):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = keras.layers.Dense(embedding_dim)

    def call(self, x):
        return tf.nn.relu(self.fc(x))


class BahdanauAttention(keras.Model):
    def __init__(self, units):
        super().__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attn = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        score = self.V(attn)
        attention_weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(attention_weights * features, axis=1)
        return context, attention_weights


class RNN_Decoder(keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super().__init__()
        self.units = units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(units, return_sequences=True, return_state=True,
                                    recurrent_initializer="glorot_uniform",
                                    reset_after=False)
        self.fc1 = keras.layers.Dense(units)
        self.fc2 = keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(units)

    def call(self, x, features, hidden):
        context, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, state = self.gru(x)
        y = self.fc1(output)
        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.fc2(y)
        return y, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class Captioner(BeamCaptioner):
    def __init__(self):
        self.word_index, self.index_word, self.vocab_size, self.max_length = \
            load_json_tokenizer(config.TOKENIZER_JSON)
        self.n_steps = self.max_length
        self.start_id = self.word_index["<start>"]
        self.end_id = self.word_index.get("<end>", -1)

        self.encoder = CNN_Encoder(config.EMBEDDING_DIM)
        self.decoder = RNN_Decoder(config.EMBEDDING_DIM, config.UNITS, self.vocab_size)

        self._build_variables()

        base = keras.applications.InceptionV3(include_top=False, weights="imagenet")
        self.feature_extractor = keras.Model(base.input, base.layers[-1].output)

        ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder)
        manager = tf.train.CheckpointManager(ckpt, config.CKPT_DIR, max_to_keep=3)
        if not manager.latest_checkpoint:
            raise RuntimeError(f"No checkpoint found in {config.CKPT_DIR}")
        self.ckpt_name = manager.latest_checkpoint
        status = ckpt.restore(self.ckpt_name)
        status.assert_existing_objects_matched()  # raises if encoder/decoder incomplete
        status.expect_partial()  # silences optimizer variables present in the file

    def _build_variables(self):
        """Blank forward pass to materialize all variables (random init)."""
        dummy = tf.zeros((1, config.ATTENTION_FEATURES_SHAPE, 2048))
        feats = self.encoder(dummy)
        hidden = self.decoder.reset_state(1)
        dec_in = tf.expand_dims([self.word_index["<start>"]], 0)
        self.decoder(dec_in, feats, hidden)

    def _features(self, img01):
        x = tf.image.resize(tf.convert_to_tensor(img01, tf.float32), (299, 299))
        x = keras.applications.inception_v3.preprocess_input(x * 255.0)  # [0,1]->[0,255]->[-1,1]
        feats = self.feature_extractor(tf.expand_dims(x, 0))
        feats = tf.reshape(feats, (feats.shape[0], -1, feats.shape[3]))
        return self.encoder(feats)

    def _init_state(self):
        return self.decoder.reset_state(1)

    def _decode_step(self, seq, features, state):
        preds, h, _ = self.decoder(tf.expand_dims([seq[-1]], 0), features, state)
        return tf.nn.log_softmax(preds[0]).numpy(), h
