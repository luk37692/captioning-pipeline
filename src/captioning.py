"""Captioning : reconstruction du tokenizer + restauration de ckpt-4 + beam search.

Le tokenizer N'EST PAS sauvegardé sur disque. On le reconstruit à l'identique en
rejouant exactement la tokenisation de l'entraînement (mêmes graines, même
sous-ensemble COCO, même TOP_K). C'est indispensable : la taille du vocabulaire
détermine les formes des couches `embedding`/`fc2` du décodeur, donc la
compatibilité avec le checkpoint.
"""
import os, re, json, collections, random
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tokenizer (reconstruit, déterministe)
# ─────────────────────────────────────────────────────────────────────────────
def _clean_caption(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "<start> " + text + " <end>"


def save_tokenizer(word_index, max_length, path=None):
    """Sauvegarde le tokenizer pour découpler l'appli de la reconstruction."""
    path = path or config.TOKENIZER_JSON
    with open(path, "w") as f:
        json.dump({"word_index": word_index, "max_length": int(max_length)}, f)


def build_tokenizer(use_saved=True):
    """Renvoie (word_index, index_word, vocab_size, max_length).

    Recharge le tokenizer sauvegardé s'il existe (use_saved), sinon le reconstruit
    à l'identique de l'entraînement (graines + sous-ensemble COCO + TOP_K).
    """
    if use_saved and os.path.exists(config.TOKENIZER_JSON):
        with open(config.TOKENIZER_JSON) as f:
            d = json.load(f)
        word_index = d["word_index"]
        index_word = {i: w for w, i in word_index.items()}
        return word_index, index_word, len(word_index), int(d["max_length"])

    with open(config.CAPTION_FILE, "r") as f:
        annotations = json.load(f)

    image_path_to_caption = collections.defaultdict(list)
    for val in annotations["annotations"]:
        path = os.path.join(config.IMAGES_DIR, config.IMG_TMPL % val["image_id"])
        image_path_to_caption[path].append(_clean_caption(val["caption"]))

    # Même séquence d'aléa qu'à l'entraînement : graine 42 fraîche avant le shuffle.
    keras.utils.set_random_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    image_paths = sorted(image_path_to_caption.keys())
    random.shuffle(image_paths)
    selected = image_paths[: config.NUM_IMAGES]

    train_captions = []
    for p in selected:
        train_captions.extend(image_path_to_caption[p])

    counter = collections.Counter(w for c in train_captions for w in c.split())
    vocab = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(config.TOP_K - 2)]
    word_index = {w: i for i, w in enumerate(vocab)}
    index_word = {i: w for w, i in word_index.items()}
    max_length = max(len(c.split()) for c in train_captions)
    return word_index, index_word, len(vocab), max_length


# ─────────────────────────────────────────────────────────────────────────────
# 2. Architecture (identique au notebook Livrable 3)
# ─────────────────────────────────────────────────────────────────────────────
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
        # reset_after=False : indispensable sur ROCm (cf. notebook Livrable 3).
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. Captioner : assemble tokenizer + modèles + checkpoint + génération
# ─────────────────────────────────────────────────────────────────────────────
class Captioner:
    def __init__(self):
        self.word_index, self.index_word, self.vocab_size, self.max_length = build_tokenizer()

        self.encoder = CNN_Encoder(config.EMBEDDING_DIM)
        self.decoder = RNN_Decoder(config.EMBEDDING_DIM, config.UNITS, self.vocab_size)

        # On matérialise les variables AVANT de restaurer : la restauration différée
        # rate l'encodeur de façon silencieuse (cf. diagnostics). Construire d'abord
        # puis restaurer garantit l'assignation des deux sous-modèles.
        self._build_variables()

        # InceptionV3 sans tête -> extracteur de features (8x8x2048 reshape 64x2048).
        base = keras.applications.InceptionV3(include_top=False, weights="imagenet")
        self.feature_extractor = keras.Model(base.input, base.layers[-1].output)

        # Checkpoint d'inférence : encodeur + décodeur uniquement (pas l'optimiseur),
        # ce qui permet d'exiger une restauration COMPLETE de ces deux objets.
        ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder)
        manager = tf.train.CheckpointManager(ckpt, config.CKPT_DIR, max_to_keep=3)
        if not manager.latest_checkpoint:
            raise RuntimeError(f"Aucun checkpoint trouvé dans {config.CKPT_DIR}")
        self.ckpt_name = manager.latest_checkpoint
        status = ckpt.restore(self.ckpt_name)
        status.assert_existing_objects_matched()  # lève si encodeur/décodeur incomplet
        status.expect_partial()  # silence les variables d'optimiseur présentes dans le fichier

    def _build_variables(self):
        """Passe à blanc pour matérialiser toutes les variables (init aléatoire)."""
        dummy = tf.zeros((1, config.ATTENTION_FEATURES_SHAPE, 2048))
        feats = self.encoder(dummy)
        hidden = self.decoder.reset_state(1)
        dec_in = tf.expand_dims([self.word_index["<start>"]], 0)
        self.decoder(dec_in, feats, hidden)

    def _features(self, x):
        feats = self.feature_extractor(x)
        feats = tf.reshape(feats, (feats.shape[0], -1, feats.shape[3]))
        return self.encoder(feats)

    @staticmethod
    def _banned_tokens(seq, n):
        """Tokens interdits à l'étape suivante pour ne pas répéter un n-gramme déjà vu."""
        if n <= 0 or len(seq) < n:
            return set()
        prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
        banned = set()
        for i in range(len(seq) - n + 1):
            if tuple(seq[i:i + n - 1]) == prefix:
                banned.add(seq[i + n - 1])
        return banned

    def caption(self, img01, beam_width=None, no_repeat_ngram_size=3):
        """Légende une image HxWx3 float [0,1] (éventuellement débruitée).

        Beam search avec blocage des n-grammes répétés : un décodeur peu entraîné
        tend à boucler ("... person up of the ..."), ce blocage l'évite.
        """
        beam_width = beam_width or config.MAX_CAPTION_BEAM
        x = tf.image.resize(tf.convert_to_tensor(img01, tf.float32), (299, 299))
        x = keras.applications.inception_v3.preprocess_input(x * 255.0)  # [0,1]->[0,255]->[-1,1]
        features = self._features(tf.expand_dims(x, 0))

        start_id, end_id = self.word_index["<start>"], self.word_index.get("<end>", -1)
        beams = [(0.0, [start_id], self.decoder.reset_state(1), False)]
        for _ in range(self.max_length):
            candidates = []
            for logp, seq, hidden, done in beams:
                if done:
                    candidates.append((logp, seq, hidden, True))
                    continue
                preds, h, _ = self.decoder(tf.expand_dims([seq[-1]], 0), features, hidden)
                lp = tf.nn.log_softmax(preds[0]).numpy()
                banned = self._banned_tokens(seq, no_repeat_ngram_size)
                taken = 0
                for t in np.argsort(lp)[::-1]:
                    t = int(t)
                    if t in banned and t != end_id:
                        continue
                    candidates.append((logp + float(lp[t]), seq + [t], h, t == end_id))
                    taken += 1
                    if taken >= beam_width:
                        break
            candidates.sort(key=lambda b: b[0] / (len(b[1]) ** 0.7), reverse=True)
            beams = candidates[:beam_width]
            if all(b[3] for b in beams):
                break

        best = max(beams, key=lambda b: b[0] / (len(b[1]) ** 0.7))
        words = [self.index_word.get(i, "<unk>") for i in best[1]]
        return " ".join(w for w in words if w not in ("<start>", "<end>", "<pad>"))
