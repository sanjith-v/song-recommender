# ---- Build stage ----------------------------------------------------
FROM python:3.11-slim AS build

# Prevent interactive prompts + speed up installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps that Annoy likes (libc++ et al.)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build Annoy index *inside* the image so we don’t ship the CSV at runtime
RUN python build_index_annoy.py \
    && rm -f spotify_data.csv          # optional: keep image slim

# ---- Runtime stage --------------------------------------------------
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# minimal runtime deps
COPY --from=build /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
