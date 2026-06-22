"""Configuration centrale de l'appli TouNum (chemins conteneur + hyperparamètres).

Tous les chemins sont ceux VUS DEPUIS LE CONTENEUR Docker, où le dossier
`Dataset/` de l'hôte est monté sur `/tf/notebooks`. Chaque valeur peut être
surchargée par une variable d'environnement de même nom.
"""
import os

def _env(name, default):
    return os.environ.get(name, default)

# CPU is forced (TOUNUM_CPU=1)
# Masks the GPU BEFORE any TensorFlow init (config is imported before tensorflow).
if _env("TOUNUM_CPU", "0") not in ("0", "", "false", "False"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
else:
    # Memory growth is forced (TOUNUM_GPU_GROWTH=1) to avoid "Failed to allocate scratch buffer" 
    # when TF tries to reserve all VRAM at once.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# Application root directory (== location of this file)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Models root (== Dataset folder of the host, mounted in the container)
TF_ROOT = _env("TF_ROOT", "/tf/notebooks")

# Modèles — TOUS embarqués dans l'appli (assets/), aucun montage externe
ASSETS = f"{_APP_DIR}/assets"
BINARY_MODEL_PATH = _env("BINARY_MODEL_PATH", f"{ASSETS}/classifier/binary_best_model.keras")
MULTI_MODEL_PATH  = _env("MULTI_MODEL_PATH",  f"{ASSETS}/classifier/doc_classifier_v9.keras")
DENOISER_PATH     = _env("DENOISER_PATH",     f"{ASSETS}/denoiser/ae_residuel_skip_best.keras")
CKPT_DIR          = _env("CKPT_DIR",          f"{ASSETS}/captioning_gru")

# Données COCO (uniquement si un tokenizer doit être reconstruit, fallback)
COCO_ROOT    = _env("COCO_ROOT", f"{TF_ROOT}/coco")
CAPTION_FILE = _env("CAPTION_FILE", f"{COCO_ROOT}/annotations/captions_train2014.json")
IMAGES_DIR   = _env("IMAGES_DIR", f"{COCO_ROOT}/train2014")
IMG_TMPL     = "COCO_train2014_%012d.jpg"

MULTI_CLASS_ORDER = ["Photo", "Painting", "Schematics", "Sketch", "Text"]
PHOTO_CLASS       = "Photo"     
CONFUSED_CLASS    = "Painting"  


BIN_THRESHOLD = float(_env("BIN_THRESHOLD", "0.5"))
# Human-in-the-loop 
MC_REVIEW_THRESH = float(_env("MC_REVIEW_THRESH", "0.60"))
BIN_REVIEW_BAND  = float(_env("BIN_REVIEW_BAND", "0.15"))

DENOISER_SIZE      = int(_env("DENOISER_SIZE", "128"))
DENOISE_MAX_PASSES = int(_env("DENOISE_MAX_PASSES", "4"))

CLASSIFIER_KIND = _env("CLASSIFIER_KIND", "cascade")
CLASSIFIERS = {
    "cascade": "Cascade multiclasse + binaire · Performant (best accuracy)",
}

# Captioning
SEED          = 42
NUM_IMAGES    = int(_env("NUM_IMAGES", "15000"))  
TOP_K         = int(_env("TOP_K", "5000"))       
EMBEDDING_DIM = 256
UNITS         = 512
ATTENTION_FEATURES_SHAPE = 64

# Cache des features InceptionV3 (.npy) et tokenizer sauvegardé (robustesse :
# l'appli le recharge au lieu de le reconstruire si le fichier existe).
FEAT_DIR       = _env("FEAT_DIR", f"{TF_ROOT}/Livrable_3_Captioning/features")
TOKENIZER_JSON = _env("TOKENIZER_JSON", f"{ASSETS}/captioning_gru/tokenizer.json")

# Ré-entraînement du décodeur : checkpoints écrits dans un dossier SEPARE
# (non destructif : ckpt-4 d'origine préservé). Pointer CKPT_DIR ici une fois validé.
TRAIN_CKPT_DIR = _env("TRAIN_CKPT_DIR", f"{TF_ROOT}/Livrable_3_Captioning/checkpoints_v2")
EPOCHS         = int(_env("EPOCHS", "40"))
BATCH_SIZE     = int(_env("BATCH_SIZE", "64"))
BUFFER_SIZE    = int(_env("BUFFER_SIZE", "1000"))

# Captioning v2 — ConvNeXt-Tiny + décodeur Transformer
# Choix du captioner : "v2plus" (EfficientNetB0 dégelé + Transformer 4 blocs, modèle
# courant, poids embarqués dans assets/), "transformer" (ConvNeXt+Transformer) ou
# "gru" (InceptionV3+GRU, historique). v2plus est autonome (aucun montage externe).
CAPTIONER_KIND = _env("CAPTIONER_KIND", "v2plus")

# Modèles de captioning proposés dans l'UI (clé interne -> libellé affiché).
# L'ordre fixe l'ordre du menu déroulant ; la 1re clé == défaut recommandé.
CAPTIONERS = {
    "v2plus":      "EfficientNetB0 + Transformer · Flickr8k/30k (embarqué)",
    "es":          "ConvNeXt-Base + Transformer · COCO (es, BLEU-4 0.32)",
    "transformer": "ConvNeXt-Tiny + Transformer · COCO+CC",
    "gru":         "InceptionV3 + GRU · COCO (historique)",
}
# Poids Keras (.weights.h5) écrits par le notebook v2 (ModelCheckpoint best/val).
CONVNEXT_CKPT  = _env("CONVNEXT_CKPT", f"{ASSETS}/captioning_convnext/best.weights.h5")
CAP_TOKENIZER_JSON = _env("CAP_TOKENIZER_JSON", f"{ASSETS}/captioning_convnext/tokenizer_v2.json")
# Vocab v2 : COCO entier, TOP_K=12000 (DOIT correspondre à l'entraînement du checkpoint).
CAP_NUM_IMAGES = int(_env("CAP_NUM_IMAGES", "82783"))
CAP_TOP_K      = int(_env("CAP_TOP_K", "12000"))
CAP_MAX_LEN    = int(_env("CAP_MAX_LEN", "40"))
# Hyperparamètres du modèle v2 (== notebook).
CAP_IMG_SIZE   = int(_env("CAP_IMG_SIZE", "224"))
CAP_FEAT_LEN   = int(_env("CAP_FEAT_LEN", "49"))     # 7x7 régions ConvNeXt-Tiny
CAP_FEAT_DIM   = int(_env("CAP_FEAT_DIM", "768"))
CAP_D_MODEL    = int(_env("CAP_D_MODEL", "512"))
CAP_NUM_LAYERS = int(_env("CAP_NUM_LAYERS", "4"))
CAP_NUM_HEADS  = int(_env("CAP_NUM_HEADS", "8"))
CAP_DFF        = int(_env("CAP_DFF", "2048"))

# Captioning v2+ — EfficientNetB0 (fine-tuné) + décodeur Transformer 4 blocs
# Modèle courant, EMBARQUÉ dans l'appli (assets/captioning/) : tokenizer picklé +
# poids encodeur/décodeur séparés. Aucune dépendance à un montage externe.
_V2P_DIR = f"{_APP_DIR}/assets/captioning"
V2P_TOKENIZER_PKL   = _env("V2P_TOKENIZER_PKL",   f"{_V2P_DIR}/tokenizer.pkl")
V2P_ENCODER_WEIGHTS = _env("V2P_ENCODER_WEIGHTS", f"{_V2P_DIR}/best_encoder.weights.h5")
V2P_DECODER_WEIGHTS = _env("V2P_DECODER_WEIGHTS", f"{_V2P_DIR}/best_decoder.weights.h5")
# Hyperparamètres (== notebook Livrable3_Captioning_v2plus, requis pour load_weights).
V2P_TOP_K       = int(_env("V2P_TOP_K", "8000"))
V2P_VOCAB_SIZE  = V2P_TOP_K + 1                 # +1 pour <pad> (index 0)
V2P_EMBEDDING_DIM = 256
V2P_NUM_HEADS   = 8
V2P_FF_DIM      = 512
V2P_NUM_LAYERS  = 4
V2P_DROPOUT     = 0.1
V2P_IMG_SIZE    = 224
V2P_FEAT_LEN    = 49        # 7x7 régions EfficientNetB0
V2P_FEAT_DIM    = 1280
V2P_GEN_MAX_LEN = int(_env("V2P_GEN_MAX_LEN", "40"))   # borne de génération
V2P_POS_MAX_LEN = int(_env("V2P_POS_MAX_LEN", "100"))  # taille de l'encodage positionnel

# Captioning es — ConvNeXt-Base (gelé) + décodeur Transformer 6 blocs
# Modèle « es » du Livrable 3 (notebook Livrable_3_Captioning_es_ConvNeXtBase).
# EMBARQUÉ : tokenizer Keras picklé + poids du décodeur (model.weights.h5).
_ES_DIR = f"{_APP_DIR}/assets/captioning_es"
ES_TOKENIZER_PKL = _env("ES_TOKENIZER_PKL", f"{_ES_DIR}/tokenizer.pkl")
ES_WEIGHTS       = _env("ES_WEIGHTS",       f"{_ES_DIR}/model.weights.h5")
# Hyperparamètres (== config.json de l'entraînement, requis pour load_weights).
ES_VOCAB_SIZE = int(_env("ES_VOCAB_SIZE", "10001"))
ES_MAX_LEN    = int(_env("ES_MAX_LEN", "22"))
ES_D_MODEL    = int(_env("ES_D_MODEL", "512"))
ES_NUM_HEADS  = int(_env("ES_NUM_HEADS", "8"))
ES_NUM_LAYERS = int(_env("ES_NUM_LAYERS", "6"))
ES_DFF        = int(_env("ES_DFF", "2048"))
ES_DROPOUT    = float(_env("ES_DROPOUT", "0.1"))
ES_IMG_SIZE   = int(_env("ES_IMG_SIZE", "224"))
ES_FEAT_LEN   = int(_env("ES_FEAT_LEN", "49"))     # 7x7 régions ConvNeXt-Base
ES_FEAT_DIM   = int(_env("ES_FEAT_DIM", "1024"))

# Runtime de l'appli
SESSIONS_DIR = _env("SESSIONS_DIR", "/app/runtime/sessions")
MAX_CAPTION_BEAM = int(_env("MAX_CAPTION_BEAM", "3"))
