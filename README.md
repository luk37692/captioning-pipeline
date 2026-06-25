# TouNum — pipeline d'annotation d'images

Application conteneurisée qui enchaîne les trois modules du projet TouNum :

1. **Tri** — cascade `multiclasse → binaire` : ne garde que les photos, avec revue
   humaine (mode strict bloquant ou auto) et reclassement manuel de toute image.
2. **Débruitage** — auto-encodeur résiduel à skip, par flag manuel sur chaque photo.
3. **Légendes** — EfficientNetB0 (fine-tuné) + Transformer 4 blocs, entraîné sur
   Flickr8k/30k, poids embarqués dans `src/assets/captioning/`. Beam search.

Interface web mobile-first (thèmes clair/sombre), servie par FastAPI ; modèles
exécutés sur GPU (AMD ROCm ou NVIDIA CUDA).

## Structure

```
.
├── docker-compose.yml         # AMD/ROCm — services app (+ profil tools)
├── docker-compose.nvidia.yml  # NVIDIA/CUDA — même app, runtime nvidia
├── Dockerfile                 # image AMD : TF/ROCm + FastAPI + code embarqué
├── Dockerfile.nvidia          # image NVIDIA : TF/CUDA + FastAPI + code embarqué
├── .env.example               # modèle de configuration (copier en .env)
├── src/                       # code applicatif servi par uvicorn
│   ├── config.py              # chemins + seuils
│   ├── pipeline.py            # cascade de classification + débruitage
│   ├── caption_common.py      # beam search partagé
│   ├── captioning_v2plus.py   # captioner EfficientNetB0 + Transformer
│   ├── models.py              # chargement paresseux + cache par modèle
│   ├── main.py                # API FastAPI
│   ├── assets/                # modèles embarqués (cf. ci-dessous)
│   ├── templates/             # page (mobile-first, thèmes clair/sombre)
│   └── static/                # css + js
└── tools/                     # hors-service web
    ├── _inspect_assets.py     # présence des assets
    └── smoke_test.py          # test du cœur (classif → débruitage → légende)
```

> **Tous les modèles sont embarqués** dans `src/assets/` (copiés dans l'image au
> build) — aucun montage externe requis pour l'inférence : `classifier/` (binaire
> + multiclasse), `denoiser/`, `captioning/` (EfficientNetB0 + Transformer).

## Configuration

```bash
cp .env.example .env
# renseigner selon le GPU :
#   AMD  : RENDER_GID, VIDEO_GID  (getent group render|video | cut -d: -f3)
#   tools: DATASET_DIR (chemin absolu, monté sur /tf/notebooks)
```

## Démarrage

```bash
# AMD / ROCm
docker compose up -d --build

# NVIDIA / CUDA  (prérequis : pilote NVIDIA + nvidia-container-toolkit)
docker compose -f docker-compose.nvidia.yml up -d --build

docker compose ps                 # état + santé
```

Accès : `http://<hôte>:8000`.

## Exploitation

```bash
docker compose logs -f app                                  # logs de l'appli
docker compose restart app                                  # redémarrage
docker compose down                                         # arrêt (volumes conservés)
docker compose --profile tools run --rm trainer             # smoke test du cœur
```

(Remplacer par `-f docker-compose.nvidia.yml` pour la variante NVIDIA.)

## Robustesse

- `restart: unless-stopped` sur l'app (repart après reboot/crash).
- Healthcheck sur `/api/health`.
