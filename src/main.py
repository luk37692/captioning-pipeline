"""Appli web TouNum : tri des photos (cascade + revue humaine) -> débruitage
manuel -> captioning. FastAPI sert l'API JSON et une page unique en 3 étapes.

Lancement (dans le conteneur) :
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os
import io
import math
import base64
import secrets
import uuid
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import pipeline
import models

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(config.SESSIONS_DIR, exist_ok=True)

app = FastAPI(title="TouNum — Pipeline photo / débruitage / captioning")

# ── Authentification basique (active uniquement si APP_PASSWORD est défini) ──────
APP_USER = os.environ.get("APP_USER", "tounum")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

@app.middleware("http")
async def basic_auth(request: Request, call_next):
    # /api/health reste public pour les sondes (healthcheck Docker, supervision).
    if APP_PASSWORD and request.url.path != "/api/health":
        hdr = request.headers.get("authorization", "")
        ok = False
        if hdr.startswith("Basic "):
            try:
                user, _, pwd = base64.b64decode(hdr[6:]).decode("utf-8").partition(":")
                ok = secrets.compare_digest(user, APP_USER) and secrets.compare_digest(pwd, APP_PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="TouNum"'})
    return await call_next(request)

app.mount("/files", StaticFiles(directory=config.SESSIONS_DIR), name="files")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

# État en mémoire (démo mono-utilisateur).
SESSIONS = {}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ─────────────────────────────────────────────────────────────────────────────
# Schémas d'API
# ─────────────────────────────────────────────────────────────────────────────
class ReviewDecision(BaseModel):
    id: str
    is_photo: bool


class ReviewBody(BaseModel):
    sid: str
    decisions: List[ReviewDecision]


class CaptionBody(BaseModel):
    sid: str


def _safe(x, ndigits=3):
    """Float compatible JSON : NaN/Inf -> 0.0 (sinon json.dumps lève une erreur).

    Un NaN ici trahit une sortie modèle instable (souvent numérique GPU/ROCm) ;
    on l'expose comme 0.0 plutôt que de planter toute la requête en 500.
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return round(x, ndigits) if math.isfinite(x) else 0.0


def _public_item(it):
    """Vue exposée au front (sans le chemin disque)."""
    return {
        "id": it["id"],
        "filename": it["filename"],
        "url": it["url"],
        "class": it["class"],
        "mc_class": it["mc_class"],
        "mc_conf": _safe(it["mc_conf"]),
        "bin_score": _safe(it["bin_score"]),
        "confidence": _safe(it["confidence"]),
        "routed": it.get("routed", False),
        "is_photo": it["is_photo"],
        "needs_review": it["needs_review"],
        "confirmed_is_photo": it["confirmed_is_photo"],
        "denoise": it["denoise"],
        "caption": it["caption"],
        "output_url": it["output_url"],
    }


def _effective_is_photo(it):
    return it["is_photo"] if it["confirmed_is_photo"] is None else it["confirmed_is_photo"]


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "templates", "index.html"))


@app.get("/api/health")
def health():
    return {"ok": True, "sessions_dir": config.SESSIONS_DIR}


@app.get("/api/captioners")
def captioners():
    """Modèles de captioning sélectionnables dans l'UI (+ défaut)."""
    return {"default": config.CAPTIONER_KIND,
            "options": [{"kind": k, "label": v} for k, v in config.CAPTIONERS.items()]}


@app.get("/api/classifiers")
def classifiers():
    """Modèles de classification sélectionnables dans l'UI (+ défaut)."""
    return {"default": config.CLASSIFIER_KIND,
            "options": [{"kind": k, "label": v} for k, v in config.CLASSIFIERS.items()]}


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Upload + (débruitage en amont, PAR IMAGE) + tri par la cascade
# Ordre logique : on débruite AVANT de classer, car les classifieurs sont
# entraînés sur des images propres. Le débruitage est choisi image par image
# (le débruiteur est lossy/128px : ne l'appliquer qu'aux images réellement bruitées).
# `denoise_flags` = liste "0"/"1" alignée sur l'ordre des fichiers envoyés.
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(
    files: List[UploadFile] = File(...),
    mode: str = Form("strict"),
    denoise_flags: str = Form(""),             # ex. "0,1,1,0" — par image, ordre des fichiers
    passes: int = Form(1),
    classifier: Optional[str] = Form(None),    # cf. /api/classifiers (une seule cascade)
    captioner: Optional[str] = Form(None),     # cf. /api/captioners (mémorisé pour l'étape légendes)
    mc_review_thresh: float = Form(config.MC_REVIEW_THRESH),
    bin_review_band: float = Form(config.BIN_REVIEW_BAND),
    bin_threshold: float = Form(config.BIN_THRESHOLD),
):
    flags = [s.strip() in ("1", "true", "True") for s in denoise_flags.split(",")] if denoise_flags else []
    try:
        clf = models.get_classifier()
        denoiser = models.get_denoiser() if any(flags) else None
    except Exception as e:
        raise HTTPException(500, f"Chargement des modèles impossible : {e}")

    passes = max(1, min(int(passes), config.DENOISE_MAX_PASSES))
    sid = uuid.uuid4().hex[:12]
    sdir = os.path.join(config.SESSIONS_DIR, sid)
    os.makedirs(sdir, exist_ok=True)

    saved = []
    for i, f in enumerate(files):
        name = os.path.basename(f.filename or "")
        if not name.lower().endswith(IMG_EXTS):
            continue
        safe = f"{len(saved):03d}_{name}".replace(os.sep, "_")
        dest = os.path.join(sdir, safe)
        with open(dest, "wb") as out:
            out.write(await f.read())
        want_denoise = flags[i] if i < len(flags) else False
        saved.append((safe, dest, name, want_denoise))

    if not saved:
        raise HTTPException(400, "Aucune image valide (jpg/png/bmp/webp) reçue.")

    # Débruitage EN AMONT (par image) : produit l'image effective (proc) servant à
    # classer PUIS à légender. Sans débruitage, l'image effective == l'originale.
    items = {}
    proc_paths = []
    for safe, dest, orig, want_denoise in saved:
        iid = uuid.uuid4().hex[:10]
        if want_denoise and denoiser is not None:
            img01 = pipeline.read_rgb01(dest)
            for _ in range(passes):
                img01 = denoiser.denoise01(img01)
            proc_name = f"{safe}.denoised.png"
            proc_path = os.path.join(sdir, proc_name)
            Image.fromarray((np.clip(img01, 0, 1) * 255).astype("uint8")).save(proc_path)
            proc_url = f"/files/{sid}/{proc_name}"
            did_denoise = True
        else:
            proc_path, proc_url = dest, f"/files/{sid}/{safe}"
            did_denoise = False
        proc_paths.append(proc_path)
        items[iid] = {
            "id": iid, "filename": orig, "path": dest,
            "url": f"/files/{sid}/{safe}",
            "proc_path": proc_path, "output_url": proc_url,
            "denoise": did_denoise, "caption": None,
        }

    # Classification sur l'image effective (débruitée si demandé).
    preds = clf.classify(proc_paths, mc_review_thresh=mc_review_thresh,
                         bin_review_band=bin_review_band, bin_threshold=bin_threshold)
    for it, pr in zip(items.values(), preds):
        it.update({
            "class": pr["class"], "mc_class": pr["mc_class"],
            "mc_conf": pr["mc_conf"], "bin_score": pr["bin_score"],
            "routed": pr["routed"], "confidence": pr["confidence"],
            "is_photo": pr["is_photo"], "needs_review": pr["needs_review"],
            "confirmed_is_photo": None,
        })

    n_denoised = sum(1 for it in items.values() if it["denoise"])
    SESSIONS[sid] = {"mode": mode, "dir": sdir, "items": items,
                     "captioner": captioner, "n_denoised": n_denoised,
                     "passes": passes,
                     "thresholds": {"mc_review_thresh": mc_review_thresh,
                                    "bin_review_band": bin_review_band,
                                    "bin_threshold": bin_threshold}}
    return {"sid": sid, "mode": mode, "n_denoised": n_denoised,
            "passes": passes, "items": [_public_item(it) for it in items.values()]}


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1bis — Revue humaine (correction des classes)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/review")
def review(body: ReviewBody):
    sess = SESSIONS.get(body.sid)
    if not sess:
        raise HTTPException(404, "Session inconnue.")
    for d in body.decisions:
        it = sess["items"].get(d.id)
        if it:
            it["confirmed_is_photo"] = d.is_photo
    return {"items": [_public_item(it) for it in sess["items"].values()]}


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Légendes (sur les photos retenues ; image déjà débruitée en amont)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/caption")
def caption(body: CaptionBody):
    sess = SESSIONS.get(body.sid)
    if not sess:
        raise HTTPException(404, "Session inconnue.")

    # En mode strict, toute image à revoir doit être tranchée avant de continuer.
    if sess["mode"] == "strict":
        unresolved = [it["filename"] for it in sess["items"].values()
                      if it["needs_review"] and it["confirmed_is_photo"] is None]
        if unresolved:
            raise HTTPException(409, f"Revue requise (mode strict) : {', '.join(unresolved)}")

    try:
        captioner = models.get_captioner(sess.get("captioner"))
    except Exception as e:
        raise HTTPException(500, f"Chargement du captioner impossible : {e}")

    # On légende TOUTES les images (catégorie + légende), pas seulement les photos.
    results = []
    for it in sess["items"].values():
        # L'image effective (proc_path) est déjà débruitée si l'option était active.
        img01 = pipeline.read_rgb01(it["proc_path"])
        it["caption"] = captioner.caption(img01)
        results.append(_public_item(it))

    return {"results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Aperçu débruitage (sans état) — débruite une image envoyée et renvoie le PNG.
# Sert à l'aperçu live « Originale vs Débruitée » dans l'écran d'options.
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/denoise-preview")
async def denoise_preview(file: UploadFile = File(...), passes: int = Form(1)):
    try:
        denoiser = models.get_denoiser()
    except Exception as e:
        raise HTTPException(500, f"Chargement du débruiteur impossible : {e}")
    passes = max(1, min(int(passes), config.DENOISE_MAX_PASSES))
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img01 = np.asarray(img, dtype=np.float32) / 255.0
    for _ in range(passes):
        img01 = denoiser.denoise01(img01)
    buf = io.BytesIO()
    Image.fromarray((np.clip(img01, 0, 1) * 255).astype("uint8")).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"passes": passes, "image": f"data:image/png;base64,{b64}"}
