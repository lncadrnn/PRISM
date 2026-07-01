# PRISM

**Progressive Realtime Identification of Synthetic Media and Disinformation**

A multimodal deep learning system that simultaneously detects AI-generated text, fake images, and deepfake videos on Filipino social media, delivered as a **Manifest V3 browser extension** (passive, real-time) and a **Next.js web app** (manual submission) with Explainable AI outputs.

> Research project for the Next Gen Start-up Competition 2026 — Mapúa University Makati (June 18, 2026).
> Authors: Lance Adrian D. Acal · Jericho G. Delos Reyes · Lee Adrian D. Noroña · Christian B. Valenzuela.

---

## Overview

PRISM scans social media content in real time and flags disinformation. Three independent forensic modules — text, image, and video — each produce a confidence score and an explainable output. These are combined at the final stage into a single credibility verdict via **late fusion**.

| Module | What it detects | Model | Explainability |
|---|---|---|---|
| **Text** | Taglish (Tagalog + English) AI-generated / fake-news captions | Fine-tuned Filipino BERT (DistilBERT-Tagalog) | LIME + Anchors — word/phrase-level highlighting |
| **Image** | GAN- and diffusion-generated / manipulated images | CNN-ViT hybrid classifier | Class Activation Maps (CAM) — heatmaps over manipulated regions |
| **Video** | Deepfakes via spatial pixel artifacts + temporal inconsistencies | Frame-level forensic engine | Frame/region-level artifact highlighting |

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  CLIENTS                                                          │
│  • Browser Extension (Manifest V3, JS)  — passive real-time scan  │
│    Facebook · TikTok · X/Twitter · Threads                        │
│  • Next.js Web App - manual content submission + interactive demo │
└──────────────────────────────┬────────────────────────────────────┘
                                │  POST /scan/extension
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  FastAPI  — AI inference layer                                    │
│  ┌──────────────────┐  ┌──────────────┐  ┌───────────────────┐    │
│  │  Text Module     │  │ Image Module │  │  Video Module     │    │
│  │  DistilBERT-     │  │ CNN-ViT      │  │  Frame-level      │    │
│  │  Tagalog         │  │ hybrid + CAM │  │  forensic engine  │    │
│  │  + LIME/Anchors  │  │              │  │  (spatial+temp.)  │    │
│  └────────┬─────────┘  └──────┬───────┘  └────────┬──────────┘    │
│           └───────────────────┴───────────────────┘               │
│                               ▼                                   │
│                  LATE FUSION (weighted average of scores)         │
│                               ▼                                   │
│               Unified credibility verdict + XAI payload           │
└──────────────────────────────┬────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  Supabase  — Postgres storage, auth, realtime, Edge Functions     │
└───────────────────────────────────────────────────────────────────┘
```

---

## Browser Extension

The extension passively scans posts on **Facebook, TikTok, X/Twitter, and Threads** and overlays a forensic verdict — without ever removing, hiding, or reporting content. PRISM is decision-support only.

### Two-axis PRISM badge

Every scanned post gets a **two-segment pill** in its top-right corner:

```
┌─────────────────────────────┐
│ ● FAKE 94%  │  ◆ AI 98%     │
└─────────────────────────────┘
  credibility     authenticity
```

PRISM reports **two orthogonal axes**:

| Axis | Question answered | Source module | Colors |
|---|---|---|---|
| **Credibility** (●) | Is the *claim* true? | `text` | FAKE → red · REAL → green |
| **Authenticity** (◆) | Is the *media* real? | `image` / `video` | AI → purple · HUMAN → blue |

A real photo can carry a false claim; an AI image can illustrate a true one. An axis with no data for a given post is hidden automatically.

**Badge states:**

| State | Appearance | Meaning |
|---|---|---|
| Scanning | `● PRISM …` (pulsing) | Post found; forensic models running |
| Verdict | Two colored segments | Scan complete; axes derived |
| Neutral | Single gray `PRISM` pill | Post detected but no analysable content |
| Error | Gray `PRISM ⚠` pill | API unreachable or scan failed |

### Hover tooltip

Hovering (or focusing) any badge opens a **shared floating card** positioned just below the badge — clamped to the viewport so it is never cut off inside `overflow:hidden` feed containers. The tooltip adapts to the badge state:

- **Scanning** — explains which modalities are being checked (caption NLP, image AI-detection, video frame extraction) and why the wait is normal.
- **Verdict** — shows per-axis detail: confidence %, LIME key-word chips (colored by direction), disinformation pattern reasons (severity-coded), image forensic signals, and a hint to open the panel for the full heatmap.
- **Error** — surfaces the API error string.

One tooltip node is shared across all badges; this avoids memory leaks from virtualized feeds that constantly attach and detach post nodes.

### Click → Visual Forensics Workspace

Clicking any badge slides in a **Shadow DOM sidebar** from the right edge of the screen. Shadow DOM isolation prevents host-page CSS from distorting the forensic UI. The panel shows:

- Verdict gauge (confidence dial)
- Scanned caption with LIME-highlighted words
- Text analysis breakdown (patterns, reasons, sources)
- CAM / GradCAM heatmap for AI-generated media
- Verified source links (wire-ready)

### Platform scanning details

| Platform | How PRISM scans | Notes |
|---|---|---|
| **Facebook** | Feed posts + Reels; Messenger, sidebar, composer, and story rails are excluded | Badge is shifted left to avoid the ⋯ menu button |
| **TikTok** | Only the **most-visible video** (IntersectionObserver, 0.5 threshold) is badged at any time | One badge follows the FYP as you scroll |
| **X / Twitter** | `article[data-testid="tweet"]` per tweet | Straightforward single-post selector |
| **Threads** | `div[data-pressable-container="true"]` with nesting guard (skips quoted posts) | Badge is shifted left to avoid the ⋯ menu button |

For **video posts** (Reels, TikTok), PRISM sends the video poster/thumbnail to the image module and the caption to the text module. Full per-frame video forensics are wired in the backend and active when the video module is loaded.

### Extension internals

```
extension/
├── manifest.json               MV3, least-privilege (activeTab, scripting, storage)
├── lib/verdict.js              window.prismVerdict — shared two-axis contract + tokens
├── ui/
│   ├── badge.js / badge.css    window.prismBadge   — two-segment pill + shared hover tooltip
│   └── sidebar.js / sidebar.css  window.prismSidebar — right forensic panel (Shadow DOM)
├── content/
│   ├── scanner.js              window.prismScanner — DOM scan + MutationObserver
│   └── main.js                 entry point: wires verdict → badge, handles SPA navigation
├── background/service-worker.js  calls the API, dedupes + caches verdicts (SHA-256 + session storage)
└── icons/
```

Content scripts load in `js`-array order and communicate via `window` globals. Per-post flow:

```
scanner.js  ──PRISM_SCAN_POST──▶  service-worker.js  ──POST /scan/extension──▶  PRISM API
   │                                      │
   │                                      ▼
main.js  ◀──PRISM_VERDICT──  chrome.tabs.sendMessage   ◀──  fused ScanResponse
   │
   ▼
badge.render()  ──hover──▶  shared tooltip
badge.render()  ──click──▶  sidebar.open()
```

**Performance / correctness details:**
- Each post is assigned a stable **FNV-1a hash** of `platform + caption + first-media-url`. Virtualized feeds (TikTok, X) that detach and reattach the same post node reuse the cached verdict instead of re-scanning.
- The service worker **dedupes in-flight requests** — two tabs scanning the same post at the same time fire one API call, not two.
- Verdicts are cached in `chrome.storage.session` (cleared on browser restart).

---

## Web App

The web front-end is a **Next.js 16** app (App Router, TypeScript) with a React 19 component set, Tailwind CSS 4, Framer Motion animations, and Lenis smooth scrolling.

**Key sections:**
- **Hero** — branching node-tree diagram illustrating the multimodal pipeline.
- **Convergence grid** — 8 AI-portrait cards that cluster toward center on scroll (spring physics).
- **Solution cards** — hover to reveal forensic telemetry, anomaly confidence, CAM region highlights.
- **Bento features** — CNN-ViT metric breakdown, late-fusion graph, module comparison.
- **Interactive demo** — submit Taglish text (LIME highlights), drag/drop an image (CAM heatmap), upload a video (temporal jitter analysis), or paste a URL (domain reputation). Full tab-based interface.
- **Extension showcase** — live badge and sidebar previews.
- **Research section + footer.**

---

## Tech Stack

| Layer | Choice |
|---|---|
| AI / ML | PyTorch / TensorFlow |
| NLP | Hugging Face Transformers |
| Vision | OpenCV |
| API layer | FastAPI + Pydantic |
| Backend | Supabase (Postgres, auth, realtime, Edge Functions) |
| Web UI | Next.js 16 · React 19 · Tailwind CSS 4 · Framer Motion · Lenis |
| Extension | JavaScript · Manifest V3 |
| Language | Python 3.10+ · TypeScript · JavaScript · SQL |

---

## Repository Layout

```
PRISM/
├── api/                      # FastAPI inference service
│   ├── main.py               # routes: /scan/text · /scan/image · /scan/video · /scan · /scan/extension
│   ├── fusion/               # weighted late-fusion logic (0.4 text · 0.35 image · 0.25 video)
│   ├── modules/
│   │   ├── text/             # DistilBERT-Tagalog + LIME/Anchors
│   │   ├── image/            # CNN-ViT hybrid + Class Activation Maps
│   │   └── video/            # frame extraction, spatial + temporal artifact detection
│   ├── schemas/              # shared VerdictResponse + ScanResponse (Pydantic)
│   └── requirements.txt
├── models/                   # trained weights (git-ignored)
├── training/                 # fine-tuning notebooks and scripts
│   ├── text/
│   ├── image/
│   └── video/
├── extension/                # Manifest V3 browser extension
│   ├── manifest.json         # targets Facebook, TikTok, X, Threads
│   ├── lib/verdict.js        # shared two-axis contract
│   ├── ui/                   # badge + sidebar (CSS + JS)
│   ├── content/              # scanner + main entry point
│   ├── background/           # service worker (API calls, deduplication, cache)
│   └── icons/
├── web/                      # Next.js 16 web app (App Router, TypeScript)
│   └── src/app/
├── data/                     # datasets (git-ignored)
├── supabase/                 # SQL migrations, Edge Functions
└── docs/                     # PRISM.pdf and architecture notes
```

---

## Getting Started

### Backend (FastAPI)

```powershell
cd api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Confirm `http://127.0.0.1:8000/health` returns `{"status":"ok"}` before loading the extension.

### Browser Extension

1. Open `chrome://extensions` → enable **Developer mode** (top-right).
2. **Load unpacked** → select the `extension/` folder.
3. Open Facebook, TikTok, X, or Threads and scroll — badges appear top-right on posts.

To point the extension at a different backend host:

```js
// Run in the service-worker DevTools console
chrome.storage.local.set({ prismApiBase: "http://192.168.1.50:8000" });
```

### Web App (Next.js)

```powershell
cd web
npm install
npm run dev
```

### Supabase

Use the Supabase CLI for local dev and migrations. Store all keys in `.env` files — never commit them.

---

## Backend API Contract

**Extension endpoint** (`POST /scan/extension`):

```json
Request:
{ "text": "caption or null", "image_urls": ["https://…jpg"], "platform": "facebook|tiktok|twitter|threads" }

Response:
{ "label": "fake|real|unknown",
  "confidence": 0.0–1.0,
  "modules": {
    "text":  { "label", "confidence", "explanation": { "summary", "reasons", "top_words", "sources" } } | null,
    "image": { "label", "confidence", "explanation": { "method", "signals", "heatmap_b64", "note" } }    | null,
    "video": { ... } | null
  },
  "explanation": { "fusion_score", "modules_used", "weights_applied" } }
```

`modules.text` drives the **Credibility** axis; `modules.image` / `modules.video` drive the **Authenticity** axis.

---

## Fusion Logic

```
verdict = 0.4 × text_conf + 0.35 × image_conf + 0.25 × video_conf
```

Only modalities present in a post are included — absent modalities are not penalised. Weights are tunable; defaults are calibrated against the evaluation set.

---

## Evaluation

- **Detection:** Precision, Recall, F1-score (primary), Accuracy — via Confusion Matrix; compared against each single-modality baseline.
- **Usability:** System Usability Scale (SUS) with social media users.
- **Latency:** real-time thresholds on consumer hardware (Intel i5 / Ryzen 5 equivalent) during active scrolling.

---

## Scope & Limitations

- Language scope is Taglish (Tagalog + English). Regional dialects (Bisaya, Ilocano) are not covered in v1; the architecture is modular for them.
- May miss zero-day synthetic media from AI models absent from training data.
- Real-time performance depends on the user's device and network.
- PRISM is a decision-support tool. Verdicts are credibility signals, not legal evidence. The system never removes, hides, downranks, or reports content.

---

## Privacy & Security

- **Least privilege:** extension requests only `activeTab`, `scripting`, `storage`, and host access to the four supported domains. No `<all_urls>`, no `tabs`, no browsing history.
- **SSRF guards:** the API only fetches public HTTP(S) URLs (blocks loopback, private ranges, reserved / multicast addresses).
- **CORS:** API accepts requests only from `chrome-extension://`, `moz-extension://`, and localhost — no wildcard.
- **Session cache:** verdicts are stored in `chrome.storage.session` and cleared when the browser restarts.

---

## Reference

Full research paper: [`docs/PRISM.pdf`](./docs/PRISM.pdf)
