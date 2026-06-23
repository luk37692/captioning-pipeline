"""Throwaway diagnostic: integrity of ALL model assets."""
import os, zipfile, pickle
import h5py
import config


def probe(path):
    if not os.path.exists(path):
        return "MISSING"
    if path.endswith(".keras"):
        try:
            z = zipfile.ZipFile(path)
            bad = []
            for n in z.namelist():
                try: z.read(n)
                except Exception as e: bad.append(f"{n}({type(e).__name__})")
            return "OK" if not bad else "BAD:" + ",".join(bad)
        except Exception as e:
            return f"ERR:{e}"
    if path.endswith((".h5",)):
        try:
            with h5py.File(path, "r") as f:
                n = [0]
                f.visititems(lambda *_: n.__setitem__(0, n[0] + 1))
            return f"OK ({n[0]} nodes)"
        except Exception as e:
            return f"BAD:{e}"
    if path.endswith(".pkl"):
        try:
            with open(path, "rb") as fh: t = pickle.load(fh)
            return f"OK ({type(t).__name__})"
        except Exception as e:
            return f"BAD:{e}"
    return "?"


ASSETS = {
    "classifier/binary": config.BINARY_MODEL_PATH,
    "classifier/multi": config.MULTI_MODEL_PATH,
    "denoiser": config.DENOISER_PATH,
    "v2plus/enc": config.V2P_ENCODER_WEIGHTS,
    "v2plus/dec": config.V2P_DECODER_WEIGHTS,
    "v2plus/tok": config.V2P_TOKENIZER_PKL,
    "es/weights": config.ES_WEIGHTS,
    "es/tok": config.ES_TOKENIZER_PKL,
    "convnext/weights": config.CONVNEXT_CKPT,
    "convnext/tok": config.CAP_TOKENIZER_JSON,
}
for label, path in ASSETS.items():
    print(f"{label:20s} {probe(path):40s} {os.path.basename(path)}")

# GRU checkpoint (TF Checkpoint, not a single file)
gru_index = os.path.join(config.CKPT_DIR, "ckpt-7.index")
print(f"{'gru/ckpt-7.index':20s} {'OK' if os.path.exists(gru_index) else 'MISSING':40s} {gru_index}")
