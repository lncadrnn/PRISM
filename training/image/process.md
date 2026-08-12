# Image module — training process

Record of how `models/image_detector.pt` (the CNN-ViT hybrid detector) got
trained, and why each step happened. Kept here so it's auditable and
repeatable — this is a log of what actually happened, not just the plan.

## Status: baseline checkpoint trained locally (CPU fallback), full-scale Kaggle GPU run blocked on a rate limit

## 1. Why not full training locally

This machine has no NVIDIA GPU — only an integrated AMD Radeon (2GB, no
CUDA). Per `training/README.md`, CPU training of the full CNN-ViT hybrid over
the full dataset is impractical (many hours+ for a run that should take
~20-40 min on a free GPU). The plan was to run the full 20-epoch job on
Kaggle instead — see section 3 for what actually happened there.

## 2. Data assembled locally, merged into `data/image/`

Two public real-vs-AI-image datasets were merged into the flat
`data/image/{real,fake}/` layout `training/image/dataset.py` expects
(duplicate filenames renamed/prefixed on collision, never overwritten):

| Source | Real | Fake | Notes |
|---|---|---|---|
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (`birdy654/...`) | 60,000 | 60,000 | 32×32, CIFAR-10 reals vs Stable Diffusion 1.4 fakes |
| [GRAVEX-200K](https://www.kaggle.com/datasets/muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal) (`muhammadbilal6305/...`) | 100,000 | 87,983 | 256×256; fake count short of 100k because the locally-downloaded archive.zip was corrupted partway through (confirmed via `unzip -t` — truncated download, not a merge bug) |
| **Total (local)** | **160,000** | **147,983** | 307,983 images |

This local copy was for inspection/organizing only, never uploaded to
Kaggle. On Kaggle both datasets were attached fresh as dataset inputs
(complete, uncorrupted — Kaggle's own copy of GRAVEX-200K has the full
100,000/100,000), so the Kaggle-side merge landed at a clean **160,000 /
160,000**.

## 3. What happened attempting the full run on Kaggle

Attempted via the `kaggle` CLI (kernel push/status/logs), authenticated with
an API token you generated and handed over for this. Ran into a chain of
infrastructure blockers, each diagnosed and fixed in turn:

1. **No internet on the kernel** — `enable_internet: true` in
   `kernel-metadata.json` was silently ignored; `git clone` failed with
   `Could not resolve host: github.com`. Root cause: Kaggle requires phone
   verification on the account before kernels get real internet access.
   Worked around it (before verification went through) by making the kernel
   fully self-contained: no git clone (training/model/dataset logic inlined
   directly in the pushed script), no live `kagglehub` download at runtime
   (both image datasets **and** the EfficientNet-B3/ViT-B16 pretrained
   weights attached as Kaggle dataset inputs — mounted read-only at
   `/kaggle/input/...`, which works with internet disabled since it's a
   Kaggle-native mount, not a network call).
2. **GPU assigned but incompatible** — once phone verification actually went
   through, `enable_gpu: true` got a real GPU, but the default assignment
   was a **Tesla P100** (`sm_60`), and the preinstalled PyTorch on Kaggle's
   image only supports `sm_70+` — GPU ops would silently fail. Fix: Kaggle's
   own kernel-metadata docs warn about exactly this and recommend explicitly
   requesting `"machine_shape": "NvidiaTeslaT4"` (not documented in `--help`,
   only in the metadata schema docs) instead of leaving it to auto-assign.
   Verified via a throwaway diagnostic kernel before touching the real one.
3. **Training hung with zero output for 10+ minutes** — root cause:
   `DataLoader(num_workers=4)` deadlocking against Kaggle's FUSE-mounted
   `/kaggle/input` datasets (a known class of issue with multiprocessing
   workers over Kaggle's mounted storage). Fixed by dropping to
   `num_workers=0` and adding per-batch progress logging so a real hang vs.
   "just slow" is immediately distinguishable next time (compare two log
   fetches a few minutes apart — identical byte-for-byte output means stuck,
   not slow).
4. **`Maximum batch GPU session count of 2 reached`** — hit after several
   kernel push iterations while debugging the above (each GPU-enabled push
   consumes a session). Deleting an old diagnostic kernel didn't free it;
   pushing a CPU-only no-op version to the same kernel slug did (superseded
   the stuck GPU session without needing a new GPU slot). Retried the real
   GPU push afterward but the limit had already re-triggered and did not
   clear within ~10 minutes of retries — looks like a short-window rate
   limit on GPU session starts, not a true concurrency cap. **Not yet
   resolved** — the full 20-epoch/307k-image run has not completed on
   Kaggle. Retry later once the limit clears (just re-run `kaggle kernels
   push -p <kernel dir>`; the script itself is already fixed and verified up
   through model init + dataset merge).

## 4. What actually produced the current checkpoint: local CPU fallback

Given the Kaggle blocker, trained a smaller baseline locally instead, to
unblock wiring up `/scan/image` now rather than waiting on Kaggle:

- Subsampled `data/image/{real,fake}` down to **1,000 real + 1,000 fake**
  (random sample, no fixed seed recorded — rerun would draw a different
  subsample).
- Ran the *actual* `training/image/train.py` (not a reimplementation) via
  `python -m training.image.train --data data/image_subset --epochs 3
  --warmup-epochs 3 --batch 16 --output models/image_detector.pt`.
  - `--epochs 3 --warmup-epochs 3` keeps the backbones frozen for the
    entire run (warmup never ends within 3 epochs) — only the fusion head
    trains. Legitimate transfer learning, just not the full fine-tune phase.
  - Two bugs surfaced and were fixed in `training/image/train.py` itself:
    - Running it as `python training/image/train.py` (as the README used
      to document) fails with `ModuleNotFoundError: No module named
      'training'` — the script's own directory ends up on `sys.path[0]`,
      not the repo root, so `from training.image.dataset import ...` can't
      resolve. Must be run as `python -m training.image.train` from the
      repo root instead. Same bug existed in the text and video trainers
      (`training/text/train.py`, `training/video/train.py` both do
      `from training.<module>.dataset import ...`) — fixed all three
      invocations in `training/README.md`.
    - `print(f"  → New best F1 ... — saved to ...")` (line ~171) crashed
      with `UnicodeEncodeError` on Windows whenever stdout isn't a live
      UTF-8 console (e.g. redirected to a log file, which is exactly what
      happens running this in the background) — the `→`/`—` characters
      don't exist in cp1252. Fixed by switching to ASCII (`->`, `--`).
      Also added per-batch progress printing (`batch i/n ... s/batch`) for
      visibility during long runs.
  - Data subsample directory was temporary (`data/image_subset/`), deleted
    after the run — the full merged set stays in `data/image/`.

### Result

```
Epoch 01/3  train_loss=0.6231  train_acc=0.6547  val_loss=0.5450  val_f1=0.7859
Epoch 02/3  train_loss=0.5055  train_acc=0.7471  val_loss=0.4812  val_f1=0.7964
Epoch 03/3  train_loss=0.4416  train_acc=0.7865  val_loss=0.4764  val_f1=0.8114

              precision    recall  f1-score   support
        real       0.86      0.64      0.74       143
        fake       0.74      0.90      0.81       157
    accuracy                           0.78       300
```

`models/image_detector.pt` (393MB) holds this checkpoint now.

## 5. Caveats / what's next

- This is a **quick baseline, not the final model**: 2,000 images (0.65% of
  the 307,983 available), 3 epochs, frozen backbones only, no end-to-end
  fine-tune phase, non-reproducible subsample (no fixed seed).
- The full run (307k+ images, 20 epochs, warmup + fine-tune) is still
  pending on Kaggle — blocked on the GPU session rate limit as of this
  writing. The kernel script itself
  (`training/image/train_kernel.py` equivalent, pushed via
  `kernel-metadata.json` with `machine_shape: NvidiaTeslaT4`,
  `num_workers=0`) is fixed and ready to resume — just needs the rate limit
  to clear and a re-push.
- `models/image_detector.pt` is what `api/modules/image/detector.py` loads
  for inference — already in place to wire up `/scan/image`, but expect it
  to be replaced once the full Kaggle run completes.
- Known dataset caveat carried forward: CIFAKE images are 32×32 (upscaled to
  224×224 at load time) — the project brief calls for higher-res
  GAN+diffusion coverage (ForenSynths/GenImage) eventually.
