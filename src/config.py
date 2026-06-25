"""Central configuration for the TouNum app (container paths + hyperparameters).

All paths are those SEEN FROM THE Docker CONTAINER, where the host `Dataset/`
folder is mounted at `/tf/notebooks`. Every value can be overridden by an
environment variable of the same name.
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

# Models — ALL embedded in the app (assets/), no external mount
ASSETS = f"{_APP_DIR}/assets"
BINARY_MODEL_PATH = _env("BINARY_MODEL_PATH", f"{ASSETS}/classifier/binary_best_model.keras")
MULTI_MODEL_PATH  = _env("MULTI_MODEL_PATH",  f"{ASSETS}/classifier/doc_classifier_v9.keras")
DENOISER_PATH     = _env("DENOISER_PATH",     f"{ASSETS}/denoiser/ae_residuel_skip_best.keras")

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
    "cascade": "Multiclass + binary cascade · best accuracy",
}

SEED = 42                          # test seed (smoke_test)

# Single captioner: v2plus = EfficientNetB0 (fine-tuné) + Transformer 4 blocs,
# entraîné sur Flickr8k/30k, poids embarqués (assets/captioning/, aucun mount).
CAPTIONER_KIND = _env("CAPTIONER_KIND", "v2plus")

# Captioning model offered in the UI (internal key -> displayed label).
CAPTIONERS = {
    "v2plus": "EfficientNetB0 + Transformer · Flickr8k/30k (embedded)",
}

# Captioning v2+ — EfficientNetB0 (fine-tuned) + 4-block Transformer decoder
# Current model, EMBEDDED in the app (assets/captioning/): pickled tokenizer +
# separate encoder/decoder weights. No dependency on an external mount.
_V2P_DIR = f"{_APP_DIR}/assets/captioning"
V2P_TOKENIZER_PKL   = _env("V2P_TOKENIZER_PKL",   f"{_V2P_DIR}/tokenizer.pkl")
V2P_ENCODER_WEIGHTS = _env("V2P_ENCODER_WEIGHTS", f"{_V2P_DIR}/best_encoder.weights.h5")
V2P_DECODER_WEIGHTS = _env("V2P_DECODER_WEIGHTS", f"{_V2P_DIR}/best_decoder.weights.h5")
# Hyperparameters (must match the embedded weights, required for load_weights).
V2P_TOP_K       = int(_env("V2P_TOP_K", "8000"))
V2P_VOCAB_SIZE  = V2P_TOP_K + 1                 # +1 for <pad> (index 0)
V2P_EMBEDDING_DIM = 256
V2P_NUM_HEADS   = 8
V2P_FF_DIM      = 512
V2P_NUM_LAYERS  = 4
V2P_DROPOUT     = 0.1
V2P_IMG_SIZE    = 224
V2P_FEAT_LEN    = 49        # 7x7 EfficientNetB0 regions
V2P_FEAT_DIM    = 1280
V2P_GEN_MAX_LEN = int(_env("V2P_GEN_MAX_LEN", "40"))   # generation bound
V2P_POS_MAX_LEN = int(_env("V2P_POS_MAX_LEN", "100"))  # positional encoding size

# App runtime
SESSIONS_DIR = _env("SESSIONS_DIR", "/app/runtime/sessions")
MAX_CAPTION_BEAM = int(_env("MAX_CAPTION_BEAM", "3"))
