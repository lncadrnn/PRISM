# Deploying the PRISM image module — what we did, why, and where it stands

Context: the web app on Vercel couldn't reach the model, because Vercel serverless
functions can't host a persistent PyTorch process (see "Why not Vercel" section
below). The plan was to run the image module as its own service on Render's free
tier and point `NEXT_PUBLIC_API_BASE` at it. **Verdict as of this log: Render's free
512MB tier cannot run this model at all — see "The wall we hit" at the bottom.**

---

## Why not just run the model on Vercel?

- Python serverless functions on Vercel cap at 500MB uncompressed on the free
  (Hobby) plan. `torch` + `transformers` + `opencv-python-headless` alone are
  close to that before adding a 376MB model weight file.
- "Large functions" (5GB) and extended timeouts exist on Vercel, but are gated to
  paid Pro/Enterprise plans.
- Serverless functions don't keep a process warm indefinitely — a ~376MB model
  would reload on cold starts, which is slow and repeats often for light traffic.
- A CPU-heavy ML endpoint would burn through Hobby's included execution/GB-hours
  quota fast, pushing you toward a paid upgrade anyway.

So: Vercel hosts the lightweight Next.js frontend (free, appropriate use), and
model inference needs a separate host built for a persistent process.

## Why Hugging Face Spaces didn't work

Historically, HF Spaces offered free CPU-Basic hosting for Docker/Gradio SDK
spaces. As of July 2026, HF eliminated free compute-backed Spaces — Docker and
Gradio SDKs now require a paid plan. Only Static Spaces (no backend compute)
remain free. Confirmed via the Spaces creation UI and HF community forum threads
from July 2026. This ruled out Spaces entirely.

## Why Render (over Koyeb, Fly.io, etc.)

Render and Koyeb were the two hosts identified with a genuinely free tier and **no
credit card required at all** (Fly.io dropped its free tier to a time-limited
trial; Cloud Run/GCP requires a billing account/card even to stay within free
tier limits). Render was picked arbitrarily between the two — both cap free
instances at 512MB RAM / 0.1 CPU, spin down after 15 min inactivity, and take
~1 min to cold-start.

---

## Step 1 — Uploading the trained model weight to Hugging Face Hub

`models/image_detector.pt` (376MB) is gitignored in the main GitHub repo (project
rule: never commit model weights). Render builds directly from GitHub, so it would
never see this file. Fix: upload it to a plain Hugging Face **model repo** (not a
Space — just free file storage, unaffected by the Spaces compute paywall), and
have the code download it at container startup.

What you ran (in your own terminal, not through Claude, so the auth token never
touched the chat):
```
hf auth login                     # pasted your HF write-scoped token
hf upload lncadrnn/prism-image-detector \
    "D:\VSC Projects\PRISM\models\image_detector.pt" image_detector.pt \
    --repo-type model
```
This created `lncadrnn/prism-image-detector` on your HF account and uploaded the
weight file. Verified afterward via HTTP HEAD request that the file's
Content-Length (393,896,839 bytes) matched the local file exactly.

**Security note during this step:** you pasted an HF token directly into the chat
at one point. That token was revoked and regenerated — never reuse a secret that
has appeared in a chat transcript, since it's logged. Always paste tokens into a
terminal's own interactive prompt, not into a chat message.

## Step 2 — Setting up the Render Web Service

Dashboard steps (dashboard.render.com/web/new):
- Connected the `lncadrnn/PRISM` GitHub repo (public, so Render could clone it
  without needing granted GitHub App access — the "we don't have access to your
  repo, but we'll try to clone it anyway" message is benign for public repos).
- Environment: **Docker** (auto-detected the root `Dockerfile`).
- Region: **Singapore** (closest to the target Filipino user base).
- Root Directory: left **blank** — the `Dockerfile` references paths like
  `api/requirements.txt` and `models/` relative to the *repo root*, so setting a
  root directory of `api` would have broken those `COPY` instructions.
- Instance Type: **Free** (512MB RAM, 0.1 CPU).
- Environment variables added:
  - `PRISM_ENABLED_MODULES=image` — see commit `c0e89ed` below.
  - `PRISM_IMAGE_MODEL_REPO=lncadrnn/prism-image-detector` — see commit `c0e89ed`.
  - (`PRISM_IMAGE_FP16=1` was added later, then removed — see commit `446d6d6`.)

---

## Commits made, in order, and why

1. **`c0e89ed` — feat: support per-module Render deployments for the API**
   Added `PRISM_ENABLED_MODULES` env var to `api/main.py` so a single deployment
   can load just one detector (e.g. only `ImageDetector`) instead of all three
   eagerly at startup — needed since one 512MB service can't hold all three
   models anyway. Added `PRISM_IMAGE_MODEL_REPO` support to
   `api/modules/image/detector.py` so it downloads the checkpoint from the HF
   Hub repo (Step 1) when the local gitignored file isn't present, via
   `huggingface_hub.hf_hub_download`.

2. **`1f46433` — fix: install CPU-only torch/torchvision in Docker build**
   The first deploy attempt installed the full CUDA/GPU build of PyTorch (pulling
   in `nvidia-cublas`, `nvidia-cudnn`, etc.) even though Render's free tier has no
   GPU — pure waste of image size and import overhead. Changed the `Dockerfile`
   to install `torch`/`torchvision` from `download.pytorch.org/whl/cpu` before the
   rest of `requirements.txt`, so those already-satisfied constraints get skipped.
   Result: build got faster and leaner, but did **not** fix the OOM (exit 137) —
   confirmed the real cost was elsewhere.

3. **`253158e` — feat: opt-in fp16 loading for the image detector** *(later reverted)*
   Hypothesis: casting the model + input to half precision would roughly halve
   the model's resident memory (376MB → 188MB). Converted the checkpoint to a
   188MB fp16 file, uploaded it to the same HF repo, and wired up
   `PRISM_IMAGE_FP16=1` to use it. Verified locally that a synthetic random-noise
   test image worked fine in fp16.
   **This was reverted in `446d6d6`** — see below.

4. **`a61ff13` — fix: skip pretrained backbone downloads when loading a fine-tuned checkpoint**
   Discovered `CNNViTHybrid.__init__` always downloaded *both* ImageNet-pretrained
   backbones (EfficientNet-B3 ~47MB + ViT-base ~330MB) before immediately
   overwriting every weight via `load_state_dict()` with your own fine-tuned
   checkpoint — pure waste, and `from_pretrained()` briefly holds both the
   downloaded weights and the model's own parameter tensors in memory at once
   (a real OOM contributor). Added a `pretrained: bool` flag to `CNNViTHybrid`;
   `ImageDetector` now passes `pretrained=False` since it always loads a full
   checkpoint immediately after construction. Training (`train.py`) is unaffected
   — it doesn't pass this flag, so it keeps the default `True` (needs real
   ImageNet init to fine-tune from).

5. **`46b9fae` — fix: make module imports conditional on PRISM_ENABLED_MODULES**
   Found that `PRISM_ENABLED_MODULES` only gated detector *instantiation* in
   `lifespan()` — the unconditional `from modules.text import TextDetector` (and
   `modules.video`) at the top of `main.py` still imported those modules' full
   dependency chains at process start regardless of which modules were actually
   enabled. For text, that chain includes `lime` → `matplotlib`, `scikit-learn`,
   `scikit-image`, `scipy` — all loaded into memory on an image-only deployment
   for nothing. Made those imports conditional on the same env var.

6. **`446d6d6` — fix: replace broken fp16 loading with meta-device + assign=True**
   Testing fp16 against a **real** JPEG (not synthetic noise) reproduced `NaN`
   logits — confirmed by running the identical image through the model in fp32
   (valid: `-30.2, 22.4`) vs fp16 (`NaN, NaN`) side by side. fp16 CPU inference is
   numerically unstable for this architecture on real images — reverted entirely,
   including deleting the fp16 loading path and the `PRISM_IMAGE_FP16` env var.
   Replaced with a different, numerically-safe technique: construct
   `CNNViTHybrid` on PyTorch's **meta device** (the ~376MB parameter skeleton
   costs zero real memory) and `load_state_dict(..., assign=True)` the
   checkpoint's tensors directly into it, instead of allocating a real fp32 model
   and then copying a separate fp32 state dict into it (that double-buffering was
   a ~750MB transient peak). Combined with `mmap=True` on the checkpoint load.
   Verified: identical logits to plain fp32 loading, and this step now only adds
   ~12MB over baseline (was the single most effective fix of everything tried).

---

## The wall we hit

After all of the above, deploys still OOM'd. Rather than guess again, profiled
actual resident memory (RSS) locally, step by step, using `psutil`:

| Stage | RSS |
|---|---|
| Python baseline | 17 MB |
| + fastapi/uvicorn | 50 MB |
| + torch (cpu) | 219 MB |
| + torchvision | 306 MB |
| + transformers | 438 MB |
| + opencv, PIL | 442 MB |
| + model construction + checkpoint load (meta+assign) | 458 MB |
| **+ one forward-only prediction (no GradCAM)** | **867 MB** |
| + one full prediction with GradCAM (forward+backward) | 1,416 MB |

Also tested the simpler pretrained-hub-only fallback (single ViT, no CNN fusion,
no GradCAM at all): baseline import 437MB → load 452MB → **one prediction: 821MB**.

**Conclusion: this isn't a loading inefficiency — it's architectural.** Just
importing torch+torchvision+transformers+opencv already consumes ~85% of Render's
512MB free-tier ceiling before any inference happens. Any real forward pass
through a ViT-scale model needs several hundred more MB for intermediate
activations, regardless of which specific model (the fused CNN-ViT or the simpler
hub fallback) is used. No further code-level optimization closes a gap this large
on this host.

## Real remaining options

1. **Run image inference from your own machine**, exposed via a free tunnel
   (Cloudflare Tunnel or ngrok) — no RAM ceiling, works with everything already
   built, zero cost, zero card. Trade-off: your machine needs to be on and the
   server running when the site needs to serve real scans.
2. **A small paid tier** (e.g. Render's Starter plan, ~2GB RAM, ~$7/month) —
   directly solves it, contradicts the "no payments" requirement you set earlier,
   but worth naming now that the free-tier wall is confirmed architectural rather
   than a bug.
3. Reduce the ML stack drastically (e.g. export to a quantized ONNX Runtime model
   instead of full PyTorch) — could plausibly fit, but is a much bigger rewrite
   with no guarantee of fitting even then; not attempted.
