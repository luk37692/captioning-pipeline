"""Captioning v2 — ConvNeXt-Tiny (encodeur gelé) + décodeur Transformer.

Modèle moderne du Livrable 3 (notebook `Livrable_3_Captioning_ConvNeXt_Transformer`).
Comme pour le captioner GRU, le tokenizer n'est pas sauvegardé : on le reconstruit à
l'identique de l'entraînement (mêmes graines, COCO entier, TOP_K=12000) — la taille du
vocabulaire fixe les dimensions des couches `embedding`/`out`, donc la compatibilité avec
les poids (`best.weights.h5`).

Interface identique à `captioning.Captioner` : `.caption(img01, beam_width=None)` où
`img01` est une image HxWx3 float dans [0, 1] (éventuellement débruitée).
"""
import os, re, json, collections, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tokenizer (reconstruit, déterministe — COCO entier, TOP_K=12000)
# ─────────────────────────────────────────────────────────────────────────────
def _clean_caption(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "<start> " + text + " <end>"


def build_tokenizer(use_saved=True):
    """Renvoie (word_index, index_word, vocab_size, max_length).

    Recharge le tokenizer sauvegardé s'il existe, sinon le reconstruit à l'identique
    de l'entraînement v2 (graine 42 + COCO entier + TOP_K=12000, max_length plafonné).
    """
    if use_saved and os.path.exists(config.CAP_TOKENIZER_JSON):
        with open(config.CAP_TOKENIZER_JSON) as f:
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

    keras.utils.set_random_seed(config.SEED); np.random.seed(config.SEED); random.seed(config.SEED)
    image_paths = sorted(image_path_to_caption.keys())
    random.shuffle(image_paths)
    selected = image_paths[: config.CAP_NUM_IMAGES]

    train_captions = []
    for p in selected:
        train_captions.extend(image_path_to_caption[p])

    counter = collections.Counter(w for c in train_captions for w in c.split())
    vocab = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(config.CAP_TOP_K - 2)]
    word_index = {w: i for i, w in enumerate(vocab)}
    index_word = {i: w for w, i in word_index.items()}
    max_length = min(max(len(c.split()) for c in train_captions), config.CAP_MAX_LEN)
    return word_index, index_word, len(vocab), max_length


def save_tokenizer(word_index, max_length, path=None):
    path = path or config.CAP_TOKENIZER_JSON
    with open(path, "w") as f:
        json.dump({"word_index": word_index, "max_length": int(max_length)}, f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Architecture (IDENTIQUE au notebook v2 — requis pour load_weights)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. Captioner : ConvNeXt + Transformer + poids + beam search
# ─────────────────────────────────────────────────────────────────────────────
class TransformerCaptioner:
    def __init__(self):
        self.word_index, self.index_word, self.vocab_size, self.max_length = build_tokenizer()

        # Encodeur visuel gelé : ConvNeXt-Tiny sans tête -> carte 7x7x768 -> 49x768.
        base = keras.applications.ConvNeXtTiny(include_top=False, weights="imagenet")
        self.feature_extractor = keras.Model(base.input, base.output)

        # Décodeur Transformer + restauration des poids (.weights.h5 de l'entraînement v2).
        self.model = CaptioningTransformer(
            self.vocab_size, config.CAP_D_MODEL, config.CAP_NUM_LAYERS,
            config.CAP_NUM_HEADS, config.CAP_DFF, self.max_length)
        # Construire les variables (1 passe à blanc) AVANT load_weights.
        dummy_feat = tf.zeros((1, config.CAP_FEAT_LEN, config.CAP_FEAT_DIM))
        dummy_in = tf.zeros((1, 2), tf.int32)
        self.model((dummy_feat, dummy_in))
        if not os.path.exists(config.CONVNEXT_CKPT):
            raise RuntimeError(f"Poids captioning v2 introuvables : {config.CONVNEXT_CKPT}")
        self.model.load_weights(config.CONVNEXT_CKPT)
        self.ckpt_name = config.CONVNEXT_CKPT

    def _features(self, img01):
        x = tf.image.resize(tf.convert_to_tensor(img01, tf.float32),
                            (config.CAP_IMG_SIZE, config.CAP_IMG_SIZE)) * 255.0
        x = keras.applications.convnext.preprocess_input(x)   # ConvNeXt: identité (attend [0,255])
        fmap = self.feature_extractor(tf.expand_dims(x, 0))
        return tf.reshape(fmap, (1, config.CAP_FEAT_LEN, config.CAP_FEAT_DIM))

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
        start_id, end_id = self.word_index["<start>"], self.word_index.get("<end>", -1)
        beams = [(0.0, [start_id], False)]
        for _ in range(self.max_length - 1):
            candidates = []
            for logp, seq, done in beams:
                if done:
                    candidates.append((logp, seq, True))
                    continue
                logits = self.model((features, tf.constant([seq], tf.int32)), training=False)
                lp = tf.nn.log_softmax(logits[0, -1]).numpy()
                banned = self._banned_tokens(seq, no_repeat_ngram_size)
                taken = 0
                for t in np.argsort(lp)[::-1]:
                    t = int(t)
                    if t in banned and t != end_id:
                        continue
                    candidates.append((logp + float(lp[t]), seq + [t], t == end_id))
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
