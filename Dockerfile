# PRISM inference API — for hosting api/ on a platform that runs a Dockerfile
# directly (e.g. Hugging Face Spaces, Docker SDK).
#
# IMPORTANT: model weights (models/*.pt) are gitignored in the main GitHub repo
# (see .gitignore) and are NOT copied by this build unless you add them to
# whatever repo/host this Dockerfile is deployed from — e.g. for a Hugging Face
# Space, add models/image_detector.pt via `git lfs` on the Space's own repo, or
# upload it through the Space's Files UI. Without it, ImageDetector silently
# falls back to the pretrained hub model (see api/modules/image/detector.py).
#
# Build/run locally to smoke-test before deploying:
#   docker build -t prism-api .
#   docker run -p 7860:7860 prism-api
FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0-0: some opencv-python-headless codecs still dlopen these
# even in "headless" builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt ./api/requirements.txt
# CPU-only torch/torchvision wheels first: the default PyPI build pulls in the
# full CUDA/cuDNN runtime (hundreds of MB, unused on free-tier hosts with no
# GPU) and its larger import footprint contributed to OOM kills on Render's
# 512MB free instance. Installed before requirements.txt so that file's
# torch>=.../torchvision>=... constraints are already satisfied and left alone.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r api/requirements.txt

COPY api/ ./api/
COPY models/ ./models/

WORKDIR /app/api

# Hugging Face Spaces (Docker SDK) expects the app on port 7860 by default.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
