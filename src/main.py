"""TouNum web app: photo sorting (cascade + human review) -> manual denoising
-> captioning. FastAPI serves the JSON API and a single 3-step page.

Launch (inside the container):
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os
import math
import uuid
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import pipeline
import models

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(config.SESSIONS_DIR, exist_ok=True)

app = FastAPI(title="TouNum — photo / denoising / captioning pipeline")

app.mount("/files", StaticFiles(directory=config.SESSIONS_DIR), name="files")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

SESSIONS = {}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ─────────────────────────────────────────────────────────────────────────────
# API Schemas
# ─────────────────────────────────────────────────────────────────────────────
class ReviewDecision(BaseModel):
    id: str
    is_photo: bool


class ReviewBody(BaseModel):
    sid: str
    decisions: List[ReviewDecision]


def _safe(x, ndigits=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return round(x, ndigits) if math.isfinite(x) else 0.0


def _public_item(it):
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



@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "templates", "index.html"))


@app.get("/api/health")
def health():
    return {"ok": True, "sessions_dir": config.SESSIONS_DIR}


@app.get("/api/captioners")
def captioners():
    """Captioning models selectable in the UI (+ default)."""
    return {"default": config.CAPTIONER_KIND,
            "options": [{"kind": k, "label": v} for k, v in config.CAPTIONERS.items()]}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1.1 — Human Review
@app.post("/api/review")
def review(body: ReviewBody):
    sess = SESSIONS.get(body.sid)
    if not sess:
        raise HTTPException(404, "Unknown session.")
    for d in body.decisions:
        it = sess["items"].get(d.id)
        if it:
            it["confirmed_is_photo"] = d.is_photo
    return {"items": [_public_item(it) for it in sess["items"].values()]}


# ─────────────────────────────────────────────────────────────────────────────
# UI contract : upload -> review -> process.
# Le tri se fait SANS débruitage (étape 1) ; le débruitage est choisi par image
# à l'étape 2 (process), juste avant la légende. Réutilise les mêmes modèles.
# ─────────────────────────────────────────────────────────────────────────────
class ProcessItem(BaseModel):
    id: str
    denoise: bool = False


class ProcessBody(BaseModel):
    sid: str
    items: List[ProcessItem]
    captioner: Optional[str] = None


@app.post("/api/upload")
async def upload(
    files: List[UploadFile] = File(...),
    mode: str = Form("strict"),
    captioner: Optional[str] = Form(None),
    mc_review_thresh: float = Form(config.MC_REVIEW_THRESH),
    bin_review_band: float = Form(config.BIN_REVIEW_BAND),
    bin_threshold: float = Form(config.BIN_THRESHOLD),
):
    try:
        clf = models.get_classifier()
    except Exception as e:
        raise HTTPException(500, f"Failed to load models: {e}")

    sid = uuid.uuid4().hex[:12]
    sdir = os.path.join(config.SESSIONS_DIR, sid)
    os.makedirs(sdir, exist_ok=True)

    saved = []
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name.lower().endswith(IMG_EXTS):
            continue
        safe = f"{len(saved):03d}_{name}".replace(os.sep, "_")
        dest = os.path.join(sdir, safe)
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved.append((safe, dest, name))

    if not saved:
        raise HTTPException(400, "No valid image (jpg/png/bmp/webp) received.")

    items = {}
    for safe, dest, orig in saved:
        iid = uuid.uuid4().hex[:10]
        items[iid] = {
            "id": iid, "filename": orig, "path": dest,
            "url": f"/files/{sid}/{safe}",
            "proc_path": dest, "output_url": f"/files/{sid}/{safe}",
            "denoise": False, "caption": None, "safe": safe,
        }

    # Classification sur images brutes (pas de débruitage au tri).
    preds = clf.classify([it["path"] for it in items.values()],
                         mc_review_thresh=mc_review_thresh,
                         bin_review_band=bin_review_band, bin_threshold=bin_threshold)
    for it, pr in zip(items.values(), preds):
        it.update({
            "class": pr["class"], "mc_class": pr["mc_class"],
            "mc_conf": pr["mc_conf"], "bin_score": pr["bin_score"],
            "routed": pr["routed"], "confidence": pr["confidence"],
            "is_photo": pr["is_photo"], "needs_review": pr["needs_review"],
            "confirmed_is_photo": None,
        })

    SESSIONS[sid] = {"mode": mode, "dir": sdir, "items": items,
                     "captioner": captioner, "n_denoised": 0, "passes": 1,
                     "thresholds": {"mc_review_thresh": mc_review_thresh,
                                    "bin_review_band": bin_review_band,
                                    "bin_threshold": bin_threshold}}
    return {"sid": sid, "mode": mode,
            "items": [_public_item(it) for it in items.values()]}


@app.post("/api/process")
def process(body: ProcessBody):
    sess = SESSIONS.get(body.sid)
    if not sess:
        raise HTTPException(404, "Unknown session.")

    if sess["mode"] == "strict":
        unresolved = [it["filename"] for it in sess["items"].values()
                      if it["needs_review"] and it["confirmed_is_photo"] is None]
        if unresolved:
            raise HTTPException(409, f"Review required (strict mode): {', '.join(unresolved)}")

    if body.captioner:
        sess["captioner"] = body.captioner
    try:
        captioner = models.get_captioner(sess.get("captioner"))
        want_denoise = any(p.denoise for p in body.items)
        denoiser = models.get_denoiser() if want_denoise else None
    except Exception as e:
        raise HTTPException(500, f"Failed to load models: {e}")

    flags = {p.id: p.denoise for p in body.items}
    sdir = sess["dir"]
    results = []
    for it in sess["items"].values():
        if it["id"] not in flags:
            continue
        if flags[it["id"]] and denoiser is not None:
            img01 = pipeline.read_rgb01(it["path"])
            img01 = denoiser.denoise01(img01)
            proc_name = f"{it['safe']}.denoised.png"
            proc_path = os.path.join(sdir, proc_name)
            Image.fromarray((np.clip(img01, 0, 1) * 255).astype("uint8")).save(proc_path)
            it["proc_path"] = proc_path
            it["output_url"] = f"/files/{body.sid}/{proc_name}"
            it["denoise"] = True
        else:
            it["proc_path"] = it["path"]
            it["denoise"] = False
        it["caption"] = captioner.caption(pipeline.read_rgb01(it["proc_path"]))
        results.append(_public_item(it))

    sess["n_denoised"] = sum(1 for it in sess["items"].values() if it["denoise"])
    return {"results": results}
