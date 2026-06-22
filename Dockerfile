# Image de l'appli TouNum : base TF/ROCm + dépendances web + code applicatif.
FROM rocm/tensorflow:latest

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
WORKDIR /app

# Code applicatif (src -> /app) et outils hors-service (tools -> /app/tools).
COPY src/ /app/
COPY tools/ /app/tools/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
