# Captioning pipeline — cleanup session

**Date:** 2026-06-23
**Goal:** kill code smells, duplication, deprecated code. Three pillars: (1) efficiency, (2) minimal complexity, (3) no fallback / no rollback.

---

## 1. Inventory (starting state)

4 captioner files = **669 lines, ~70% copy-paste**.

### Pillar 1 — Efficiency (waste)

| Block | Copies | Files |
|---|---|---|
| `_banned_tokens` | 4 identical | all captioners |
| `caption()` beam loop | 4 near-identical | all captioners |
| `positional_encoding` | 2 identical | v2plus, es |
| `DecoderLayer` | 3 (2 identical + 1 variant) | v2plus, es, transformer |
| `TransformerDecoder` | 2 near-identical (only default hyperparams differ) | v2plus, es |
| `_clean_caption` + `build_tokenizer` + `save_tokenizer` | 2 near-identical | captioning, transformer |

Beam search + n-gram blocker + tokenizer = whole machinery pasted 4×. Already drifting (transformer masks `<start>` only, v2plus/es mask `<unk>`, gru carries GRU state) → bug farm.

**Dead-on-disk:** committed weights exist only for `gru`. `v2plus` (default), `es`, `transformer` weights gitignored + absent (user has them at home, adds later).

### Pillar 2 — Minimal complexity (over-build)
- 4 captioners, all exposed in UI.
- `config.py` 151 lines, ~40 `_env()` overrides — including hyperparams locked to trained weights (fake flexibility: change one → `load_weights` crashes).
- `build_tokenizer` rebuild-from-COCO branch (~25 lines ×2) — never runs in prod since tokenizer JSON ships embedded.

### Pillar 3 — No fallback / no rollback (rule violated)
- `build_tokenizer(use_saved=True)` — load-or-rebuild fallback.
- `get_captioner` — unknown kind silently falls back to default (twice).
- `_safe()` — swallows bad floats → 0.0.
- `status.expect_partial()` — silences missing-variable mismatches.

---

## 2. Decisions (user)
- **Keep all 4 captioners.** `es`/`transformer` weights added later from home.
- **Kill silent fallbacks** — fail loud, no bricolage.
- **No reference to COCO anywhere in the code.**

---

## 3. Changes applied

### Dedup → shared base
New `src/caption_common.py`. Holds the machinery that was pasted 4×:
- `_banned_tokens`, beam-search `caption()` template (subclass provides `_features` + `_decode_step` + opaque `state` hook → stateful GRU and 3 stateless Transformers share one loop)
- `positional_encoding`, `SinusoidalDecoderLayer`, `SinusoidalTransformerDecoder` (v2plus/es shared)
- `load_json_tokenizer`

4 captioners → thin subclasses of `BeamCaptioner`.

| File | before | after |
|---|---|---|
| captioning.py | 195 | 105 |
| captioning_es.py | 142 | 49 |
| captioning_v2plus.py | 146 | 53 |
| captioning_transformer.py | 186 | 100 |
| caption_common.py | — | ~140 |

One beam search, one n-gram blocker, one sinusoidal Transformer. Bug fix = 1 edit.

### Fail-loud (pillar 3)
- `get_captioner("bogus")` → `RuntimeError` instead of silent default swap.
- Missing tokenizer JSON at inference → `RuntimeError` instead of silent COCO rebuild.
- **Kept** `_safe()` (JSON can't encode NaN/inf — serialization guard, not a model fallback) and `expect_partial()` (silences optimizer slots only; `assert_existing_objects_matched()` above it already fails loud). Not bricolage.

### COCO purge
- **Deleted** `tools/train_captioning.py` (100% COCO retraining; weights now external).
- `build_json_tokenizer` (load-or-rebuild-from-COCO) → `load_json_tokenizer` (load-or-raise). Dropped `clean_caption`, `save_json_tokenizer`, imports `re`/`collections`/`random`.
- `config.py`: removed `COCO_ROOT`, `CAPTION_FILE`, `IMAGES_DIR`, `IMG_TMPL`, `NUM_IMAGES`, `TOP_K`, `FEAT_DIR`, `TRAIN_CKPT_DIR`, `EPOCHS`, `BATCH_SIZE`, `BUFFER_SIZE`, `CAP_NUM_IMAGES`, `CAP_TOP_K`, `CAP_MAX_LEN`. Stripped "COCO" from UI labels + comments. Kept `SEED` (smoke_test RNG only).
- `smoke_test.py`: "sans COCO" → "sans dataset".

`max_length` now read from the tokenizer JSON (was capped by `CAP_MAX_LEN`).

---

## 4. Verification
- `grep -i coco` over `src/` + `tools/` = **NONE** (assets excluded).
- All `.py` compile (`py_compile`).
- AST check: 4 captioners subclass `BeamCaptioner`, expose `_features` + `_decode_step`.
- No dead `config.*` refs remain.
- **Not runtime-tested** — no TensorFlow locally (Docker-only). Run `tools/smoke_test.py` in the container to confirm end-to-end.

## 5. Open items
- `README.md` still mentions COCO (docs, not code) — left as-is unless asked.
- Training reproducibility for `es`/`transformer`/`gru` now lives only in home notebooks (deliberate, per "no COCO" instruction).
