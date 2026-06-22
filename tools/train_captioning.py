"""Ré-entraînement (non destructif) du décodeur GRU de captioning sur COCO.

Reprend l'architecture du captioner historique (`captioning.CNN_Encoder` +
`RNN_Decoder`, attention de Bahdanau / show-attend-tell) et son tokenizer reconstruit,
puis ré-entraîne le décodeur sur les features InceptionV3 de COCO. Les checkpoints sont
écrits dans `config.TRAIN_CKPT_DIR` (dossier SÉPARÉ : le ckpt d'origine reste intact) et
le tokenizer est sauvegardé pour découpler l'inférence de la reconstruction COCO.

Lancement (profil tools, COCO monté via DATASET_DIR) :

    docker compose --profile tools run --rm trainer

Prérequis : `config.CAPTION_FILE` et `config.IMAGES_DIR` accessibles (COCO train2014).
"""
import os
import sys
import time
import collections
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Les modules applicatifs vivent dans /app (conteneur) ou src/ (local).
HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in ("/app", os.path.join(os.path.dirname(HERE), "src")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

import config        # noqa: E402
import captioning    # noqa: E402  (CNN_Encoder, RNN_Decoder, build_tokenizer, _clean_caption)

AUTOTUNE = tf.data.AUTOTUNE


# ─────────────────────────────────────────────────────────────────────────────
# 1. Données COCO : (chemin image -> légendes), aligné sur build_tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def load_coco_pairs():
    """Renvoie (img_paths, captions) alignés sur le sous-ensemble d'entraînement."""
    import json
    with open(config.CAPTION_FILE, "r") as f:
        annotations = json.load(f)

    image_path_to_caption = collections.defaultdict(list)
    for val in annotations["annotations"]:
        path = os.path.join(config.IMAGES_DIR, config.IMG_TMPL % val["image_id"])
        image_path_to_caption[path].append(captioning._clean_caption(val["caption"]))

    # Même séquence d'aléa que build_tokenizer (graine 42) pour rester cohérent.
    keras.utils.set_random_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    image_paths = sorted(image_path_to_caption.keys())
    random.shuffle(image_paths)
    selected = image_paths[: config.NUM_IMAGES]

    img_paths, captions = [], []
    for p in selected:
        for c in image_path_to_caption[p]:
            img_paths.append(p)
            captions.append(c)
    return img_paths, captions


# ─────────────────────────────────────────────────────────────────────────────
# 2. Features InceptionV3 (cache .npy dans config.FEAT_DIR)
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_extractor():
    base = keras.applications.InceptionV3(include_top=False, weights="imagenet")
    return keras.Model(base.input, base.layers[-1].output)


def _feat_path(img_path):
    return os.path.join(config.FEAT_DIR, os.path.basename(img_path) + ".npy")


def cache_features(unique_paths, extractor, batch=16):
    """Extrait et met en cache (8x8x2048 -> 64x2048) les features manquantes."""
    os.makedirs(config.FEAT_DIR, exist_ok=True)
    todo = [p for p in unique_paths if not os.path.exists(_feat_path(p))]
    print(f"Features à extraire : {len(todo)} / {len(unique_paths)}")

    def load(p):
        img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
        img = tf.image.resize(img, (299, 299))
        img = keras.applications.inception_v3.preprocess_input(tf.cast(img, tf.float32))
        return img, p

    ds = tf.data.Dataset.from_tensor_slices(todo).map(load, num_parallel_calls=AUTOTUNE).batch(batch)
    for imgs, paths in ds:
        feats = extractor(imgs)
        feats = tf.reshape(feats, (feats.shape[0], -1, feats.shape[3]))
        for f, p in zip(feats.numpy(), paths.numpy()):
            np.save(_feat_path(p.decode("utf-8")), f)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset (features, séquence de tokens)
# ─────────────────────────────────────────────────────────────────────────────
def encode_caption(caption, word_index, max_length):
    ids = [word_index.get(w, word_index["<unk>"]) for w in caption.split()]
    ids = ids[:max_length]
    return ids + [word_index["<pad>"]] * (max_length - len(ids))


def make_dataset(img_paths, captions, word_index, max_length):
    seqs = np.array([encode_caption(c, word_index, max_length) for c in captions], np.int32)
    feat_paths = [_feat_path(p) for p in img_paths]

    def load(feat_path, seq):
        feat = tf.numpy_function(lambda fp: np.load(fp.decode("utf-8")), [feat_path], tf.float32)
        feat.set_shape((config.ATTENTION_FEATURES_SHAPE, 2048))
        return feat, seq

    return (tf.data.Dataset.from_tensor_slices((feat_paths, seqs))
            .shuffle(config.BUFFER_SIZE)
            .map(load, num_parallel_calls=AUTOTUNE)
            .batch(config.BATCH_SIZE).prefetch(AUTOTUNE))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Boucle d'entraînement (teacher forcing, perte masquée sur <pad>)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    word_index, index_word, vocab_size, max_length = captioning.build_tokenizer(use_saved=False)
    print(f"Vocab={vocab_size}  max_length={max_length}")
    captioning.save_tokenizer(word_index, max_length)  # découple l'inférence de COCO

    img_paths, captions = load_coco_pairs()
    extractor = build_feature_extractor()
    cache_features(sorted(set(img_paths)), extractor)
    dataset = make_dataset(img_paths, captions, word_index, max_length)

    encoder = captioning.CNN_Encoder(config.EMBEDDING_DIM)
    decoder = captioning.RNN_Decoder(config.EMBEDDING_DIM, config.UNITS, vocab_size)
    optimizer = keras.optimizers.Adam()
    loss_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    pad_id = word_index["<pad>"]
    start_id = word_index["<start>"]

    def masked_loss(real, pred):
        mask = tf.cast(tf.not_equal(real, pad_id), tf.float32)
        loss = loss_obj(real, pred) * mask
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)

    @tf.function
    def train_step(feats, target):
        loss = 0.0
        hidden = decoder.reset_state(tf.shape(target)[0])
        dec_in = tf.expand_dims([start_id] * tf.shape(target)[0], 1)
        with tf.GradientTape() as tape:
            features = encoder(feats)
            for t in range(1, target.shape[1]):
                preds, hidden, _ = decoder(dec_in, features, hidden)
                loss += masked_loss(target[:, t], preds)
                dec_in = tf.expand_dims(target[:, t], 1)  # teacher forcing
        total = loss / tf.cast(target.shape[1], tf.float32)
        variables = encoder.trainable_variables + decoder.trainable_variables
        optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return total

    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, config.TRAIN_CKPT_DIR, max_to_keep=3)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"Reprise depuis {manager.latest_checkpoint}")

    for epoch in range(config.EPOCHS):
        t0 = time.time()
        running = 0.0
        steps = 0
        for feats, target in dataset:
            running += float(train_step(feats, target))
            steps += 1
        path = manager.save()
        print(f"Epoch {epoch + 1}/{config.EPOCHS}  loss={running / max(steps, 1):.4f}  "
              f"{time.time() - t0:.0f}s  ckpt={path}")

    print(f"\nEntraînement terminé. Checkpoints dans {config.TRAIN_CKPT_DIR}")
    print("Pointer config.CKPT_DIR vers ce dossier une fois la qualité validée.")


if __name__ == "__main__":
    main()
