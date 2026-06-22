# TouNum — pipeline d'annotation d'images

Application conteneurisée qui enchaîne les trois modules du projet TouNum :

1. **Tri** — cascade `multiclasse → binaire` : ne garde que les photos, avec revue
   humaine (mode strict bloquant ou auto) et reclassement manuel de toute image.
2. **Débruitage** — auto-encodeur résiduel à skip, par flag manuel sur chaque photo.
3. **Légendes** — modèle choisi dans l'UI parmi : EfficientNetB0 (fine-tuné) +
   Transformer 4 blocs (**défaut, poids embarqués** dans `src/assets/captioning/`),
   ConvNeXt-Tiny + Transformer, ou InceptionV3 + GRU (historique). Beam search.

Interface web mobile-first (thèmes clair/sombre), servie par FastAPI ; modèles
exécutés sur GPU (ROCm) ; accès privé via Tailscale.

## Structure

```
.
├── docker-compose.yml     # services : app + tailscale (+ profil tools)
├── Dockerfile             # image : TF/ROCm + FastAPI + code embarqué
├── .env.example           # modèle de configuration (copier en .env)
├── src/                   # code applicatif servi par uvicorn
│   ├── config.py          # chemins + seuils
│   ├── pipeline.py        # cascade de classification + débruitage
│   ├── captioning.py      # captioner GRU (InceptionV3 + GRU)
│   ├── captioning_transformer.py  # captioner ConvNeXt-Tiny + Transformer
│   ├── captioning_v2plus.py       # captioner EfficientNetB0 + Transformer (défaut)
│   ├── models.py          # chargement paresseux + cache par modèle
│   ├── main.py            # API FastAPI + authentification
│   ├── assets/            # TOUS les modèles embarqués (cf. ci-dessous)
│   ├── templates/         # page (mobile-first, thèmes clair/sombre)
│   └── static/            # css + js
└── tools/                 # hors-service web
    ├── train_captioning.py
    └── smoke_test.py
```

> **Tous les modèles sont embarqués** dans `src/assets/` (copiés dans l'image au
> build) — aucun montage externe requis pour l'inférence :
> `classifier/` (binaire + multiclasse), `denoiser/`, `captioning/` (EfficientNetB0
> + Transformer, défaut), `captioning_convnext/`, `captioning_gru/`.
> Seules les **données** (COCO, features `.npy`) restent hors dépôt, montées sur
> `/tf/notebooks` via `DATASET_DIR` pour le profil `tools` (ré-entraînement).

## Configuration

```bash
cp .env.example .env
# renseigner : APP_PASSWORD, RENDER_GID, VIDEO_GID, DATASET_DIR
#   GID GPU :  getent group render | cut -d: -f3   /   getent group video | cut -d: -f3
#   TS_AUTHKEY : seulement au tout premier enregistrement du nœud Tailscale
```

## Démarrage

```bash
docker compose up -d --build
docker compose ps                 # état + santé
```

Accès depuis un appareil du tailnet : `http://tounum:8000`
(authentification `APP_USER` / `APP_PASSWORD`).

## Exploitation

```bash
docker compose logs -f app                                  # logs de l'appli
docker compose restart app                                  # redémarrage
docker compose down                                         # arrêt (volumes conservés)
docker compose --profile tools run --rm trainer             # ré-entraîner le décodeur
docker compose --profile tools run --rm trainer python tools/smoke_test.py   # test du cœur
```

## Robustesse

- `restart: unless-stopped` sur l'app et Tailscale (repartent après reboot/crash).
- Healthcheck sur `/api/health` (public, exempt d'auth).
- État Tailscale dans un volume dédié : nœud stable, pas de ré-authentification.
- Accès uniquement via le réseau privé Tailscale, protégé par mot de passe.
