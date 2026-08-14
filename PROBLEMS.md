# PRISM — open problems

Everything below is unresolved as of the 2026-08-14 repo audit — either
flagged as "unresolved" by the cleanup pass that followed the audit, or a
medium/high-severity finding that pass correctly left alone because fixing it
requires a product, security, or architecture decision rather than a
mechanical edit. Organized by area. Each item: what's wrong, why it matters,
what to do about it.

See also `loopholes.md` for the deeper conceptual gaps (fusion averaging
orthogonal signals, video module not actually wired into the extension's
real-time path, domain-mismatch in training data) — this file is the flatter
audit-findings list; that one is the "is the core idea sound" read.

---

## 0. Most pressing right now: the Render.com free-tier deployment wall

The image module's API can't be hosted on Render's free tier — importing
`torch`+`torchvision`+`transformers`+`opencv` alone consumes ~85% of the
512MB RAM ceiling before any inference runs, and this is confirmed
architectural (profiled RSS at each stage), not a bug to be optimized away.
Full story, commit-by-commit history, and the memory-profiling table are in
`docs/render-deployment-log.md` — not repeated here.

**The three real options, unchanged from that log:**
1. **Self-hosted tunnel** (Cloudflare Tunnel / ngrok) from your own machine —
   zero cost, zero card, no RAM ceiling, works with everything already built.
   Trade-off: your machine has to be on and the server running whenever the
   site needs to serve a real scan.
2. **A small paid tier** (e.g. Render Starter, ~2GB RAM, ~$7/month) — directly
   solves it, but contradicts the "no payments" constraint set earlier.
3. **A lighter ML runtime** (e.g. quantized ONNX Runtime instead of full
   PyTorch) — could plausibly fit in 512MB, but is a materially bigger
   rewrite with no guarantee of fitting even after the work is done; not
   attempted yet.

This blocks any public deployment of the image module (and, by extension,
the video module, which has the same dependency footprint) until one of the
three is chosen. Recommend deciding this before investing more time in
feature work that assumes a live hosted API.

---

## 1. Frontend (web app)

### 1.1 No text/caption input anywhere in the web app (high)
`web/src/app/page.tsx`'s workspace only has "Photo Forensic" and "Video
Tracker" tabs (lines ~1288-1291). There is no text/caption tab, even though
`POST /scan/text` is a fully working backend endpoint with LIME/Anchors
explanations (`api/modules/text/explainer.py`), the extension already renders
this exact XAI output (`extension/ui/sidebar.js` `renderTextAnalysis`/
`highlightCaption`), and CLAUDE.md's build order puts text first because it's
"most labeled data available; fastest to validate." The web app — the
project's second delivery surface for manual submission — doesn't exercise
its most-validated module at all.
**Recommendation:** add a Text/Caption tab that POSTs to `${API_BASE}/scan/text`
and renders label/confidence plus word-level highlights, mirroring
`sidebar.js`'s existing rendering logic.

### 1.2 Video tab is a hardcoded fake, not wired to `/scan/video` (high)
Both `handleGlobalDrop`'s video branch (lines ~212-241) and
`handleDemoVerify`'s video branch (lines ~543-561) use `setTimeout()` and
fabricate a verdict — hardcoded `confidence`, `jitterFrames: [3, 5]`,
`syncGap: "24ms"`. The code self-documents this at line ~543: `// Video
module isn't wired to the API yet — keeps the illustrative demo.` Meanwhile
`POST /scan/video` (`api/main.py` lines ~358-387) is a real, working endpoint
backed by ELA + optical-flow forensics (`api/modules/video/detector.py`). Any
visitor uploading a video currently gets a canned answer regardless of the
file's actual content.
**Recommendation:** wire the video tab's upload/drop handler to POST to
`${API_BASE}/scan/video` the same way `scanPhoto()` already does for images,
and render the real explanation instead of the fabricated jitter/sync values.

### 1.3 Dead CTA: "Add to Chrome (Free)" links to a nonexistent anchor (medium)
Links to `href="#download"` (line ~1682); no element with `id="download"`
exists anywhere in the page (the only in-page anchors are `#solutions`,
`#bento`, `#extension`, `#publications`, `#demo`). This is the page's primary
CTA and it's currently a no-op click.
**Recommendation:** either add a real `#download` section (install
instructions or a Chrome Web Store link once published), or point the button
at `#extension` like the other install CTAs already do — a judgment call on
intended destination for the project owner.

### 1.4 "Measured" metrics on the public page are invented, not measured (medium)
The Bento section shows a hardcoded `F1-Score // 96.4%` badge and an
inference-latency counter that animates up to a fixed 412ms, captioned
"Measured Inference Latency" / "Under 2.0s Threshold." No evaluation harness
(confusion matrix, F1-vs-baseline, latency percentiles — build step 9) exists
anywhere in the repo to have produced these numbers. This is a public-facing
academic-competition page presenting unverified figures as measured fact.
**Recommendation:** either relabel as illustrative/target figures, or hold
off until the real evaluation harness exists and has actually been run —
this is a messaging decision, not a code fix, so it needs the project
owner's call.

### 1.5 Web app's "manual submission" surface is image-only (medium)
Overlaps with 1.1/1.2: `web/src/app/page.tsx` only calls `POST /scan/image`
anywhere in `web/src`. There are no calls to `/scan/text`, `/scan/video`, or
the fused `/scan`/`/scan/extension` endpoints. The north-star's second
delivery surface currently functions as an image-only demo widget, not a
general multimodal manual-submission tool.
**Recommendation:** resolve alongside 1.1 and 1.2; alternatively, explicitly
document the web app's current scope as "image-only demo, full multimodal
submission pending" so the gap between plan and implementation is visible
rather than implied.

---

## 2. Backend / ML modules

### 2.1 Text module backbone diverges from the documented plan (high)
CLAUDE.md specifies DistilBERT-Tagalog, fine-tuned in-project on Vera
Files/AFP + FakeNewsNet, compressed via knowledge distillation for
lightweight real-time inference. Production (`api/modules/text/model.py`
`MODEL_ID`, `api/modules/text/detector.py`, `training/text/train.py`'s
default `--model-id`) actually runs the unmodified third-party hub checkpoint
`iceman2434/xlm-roberta-base-fake-news-detection-tl` — an XLM-RoBERTa-**base**
model, ~3x larger than a distilled model, never fine-tuned by this project.
`training/text/train.py` itself warns that a checkpoint trained on a
different backbone isn't loadable by the current inference `TextDetector` —
the divergence is known in the code, just not reflected in the docs. XLM-R-
base's size also works against the <2s real-time latency target on
i5/Ryzen5-class hardware.
**Recommendation:** either update CLAUDE.md to name the actual production
backbone and drop the DistilBERT-Tagalog claim, or invest in the real
fine-tune/distillation on Vera Files/AFP-labeled Taglish data and measure
latency against the 2s budget before shipping. Architectural decision, left
for the project owner.

### 2.2 Video module has no trained model; calibration claims are unverified (high)
`models/video/` never held a real checkpoint, so `VideoDetector` always runs
the rule-based path: ELA + noise-residual (`SpatialAnalyzer`) and Farneback
optical-flow jitter/lip-sync heuristics (`TemporalAnalyzer`) in
`api/modules/video/forensics.py`. Docstrings/comments there claim thresholds
(`_ela_low=2.0`, `_ela_high=15.0`, etc.) are "empirically calibrated on
FaceForensics++," but no calibration script, notebook, or dataset exists
anywhere in the repo backing that claim — `data/video/` is also just
`.gitkeep`. The video module's real F1/precision/recall against the plan's
≥90% F1 target have never been measured.
**Recommendation:** either train and ship a real `video_detector.pt`
checkpoint and validate the rule-based thresholds against a labeled dataset,
or correct the docstrings to stop asserting a calibration that doesn't exist.
Left for the project owner to prioritize.

### 2.3 Evaluation harness (build step 9) is entirely missing (high)
No confusion-matrix script, F1-vs-single-modality baseline comparison,
latency stress-test harness, or SUS survey artifact exists in `api/` or
`training/` — only per-epoch `f1_score` calls inside individual training
loops, which measure one module in isolation, never the fused system against
baselines. Given the stated competition deadline (June 18, 2026) is ~2
months past, this step appears not to have been started.
**Recommendation:** build a script that runs the fused `/scan` endpoint and
each single-modality endpoint over a held-out labeled set, computes
precision/recall/F1/accuracy via confusion matrix, measures p50/p95 latency
on consumer-class hardware (≥100 instances/modality per the plan's
requirements), plus a plan for administering the SUS survey (≥30
participants). Substantial new-feature build, not a cleanup-scope fix.

### 2.4 Video module XAI has no frame/region-level overlay (medium)
CLAUDE.md specifies video XAI as "frame/region-level artifact overlay."
`VideoDetector.predict()` only returns scalar `spatial_score`/
`temporal_score`/`frames_analyzed` — no heatmap, frame overlay, or region
highlight anywhere in `api/modules/video/`. `sidebar.js`'s
`renderMediaAnalysis` only renders a heatmap when `explanation.heatmap_b64`
is present, which the video module never sets (only the image module does,
via GradCAM). A video verdict in the UI shows two raw numbers with no visual
explanation, unlike image's CAM heatmap or text's LIME highlighting.
**Recommendation:** add a per-frame or region-level overlay to
`VideoDetector`'s explanation (flag the sampled frame(s) with the highest
ELA/jitter score, return an annotated frame/crop), and extend `sidebar.js` to
render it. Feature addition, left for a dedicated pass.

### 2.5 Text module XAI claims "LIME + Anchors"; only LIME is implemented (medium)
CLAUDE.md, `README.md`, and `extension/README.md` all describe the text
module's explainability method as "LIME + Anchors" (word/phrase-level
highlighting). `api/modules/text/explainer.py` only implements
`LIMETextExplainer` plus a rule-based Filipino pattern analyzer
(`patterns.py`); its returned `explanation.method` is literally
`"pattern+LIME"`. There is no Anchors implementation anywhere in `api/` —
`grep -rin anchor api/` matches nothing outside this docs claim. Same shape
as 2.1/2.2: a plan that names two techniques, and an implementation that
ships only one.
**Recommendation:** either implement an Anchors explainer alongside LIME
(the two are complementary — LIME gives word weights, Anchors gives
if-then decision rules), or update CLAUDE.md/README.md/extension/README.md
to say "LIME" only until Anchors is actually built. Feature-vs-docs
decision, left for the project owner.

---

## 3. Security / reliability (QA)

### 3.1 SSRF guard bypass via redirect following + DNS-rebinding TOCTOU (high)
`_is_public_http_url()` (`api/main.py`, used by `/scan/extension`) resolves
and validates the URL's *original* hostname, but the actual fetch uses
`httpx.AsyncClient(..., follow_redirects=True)` and never re-validates the
redirect target. A public URL that later 302s to `169.254.169.254/latest/
meta-data/`, `localhost:<internal-port>`, or any RFC1918 address fully
defeats the SSRF guard — reachable via `/scan/extension`'s user-supplied
`image_urls`, so by anyone who can get PRISM to scan a crafted post/page.
There's also a separate DNS-rebinding TOCTOU gap: the guard's own DNS lookup
and `httpx`'s later independent lookup at connect-time can resolve
differently for a short-TTL record.
**Recommendation:** disable `follow_redirects` and manually validate each
`Location` header hop against `_is_public_http_url()` before following it
(cap the number of hops), and pin the connection to the specific IP that was
validated (custom transport/socket options) to close the rebinding window.
Security-sensitive change requiring careful testing before shipping.

### 3.2 Rate-limit bypass via spoofable `X-Forwarded-For` + unbounded memory growth (high)
`_client_ip()` (`api/main.py`) trusts client-supplied `X-Forwarded-For`
unconditionally, with no check that the request passed through a trusted
proxy, and uses the left-most (attacker-controlled) value as the rate-limit
bucket key. Any caller can spoof a new "IP" per request to bypass the 20
req/min cap entirely. This also leaks memory: `_request_log` (a
`defaultdict(deque)`) gets a new entry per spoofed key that's never evicted,
which is a real concern given this project's already-documented OOM
sensitivity on constrained hosting.
**Recommendation:** only trust `X-Forwarded-For`/`X-Real-IP` from a known
trusted proxy (allowlist the hosting platform's proxy IPs, or use the
platform's dedicated client-IP header), take the right-most untrusted-free
hop, and periodically prune empty deques from `_request_log` (or cap/LRU it).
Needs a decision on trusted-proxy strategy specific to the eventual hosting
platform — still undecided per problem 0 above.

### 3.3 No automated tests or CI anywhere in the repo (high)
Zero test files, no pytest config/`conftest.py`, no GitHub Actions workflow
for `api/`, `web/`, or `extension/` (confirmed via `git ls-files` and search
for `pytest.ini`/`.github/workflows`). `.gitignore` also excludes
`api/smoke_*.py` and `api/test_*.py`, so even ad hoc local sanity scripts
never become committed regression guards. Nothing currently catches a
regression in fusion logic, schema validation, SSRF/rate-limit behavior, or
any detector before it ships.
**Recommendation:** add a minimal pytest suite covering `fuse()` edge cases
(no modalities, all abstain, single modality), `/scan*` routes via FastAPI's
`TestClient` (malformed JSON, oversized upload, wrong content-type,
SSRF-guard unit tests with mocked DNS), and wire a basic GitHub Actions
workflow to run it on push/PR. Also reconsider the `test_*.py`/`smoke_*.py`
gitignore exclusion — move real tests under a `tests/` directory so they get
committed. Substantial engineering investment, not a mechanical fix.

### 3.4 Blocking synchronous model inference inside async route handlers (medium)
All `/scan*` routes in `api/main.py` are `async def` but call the detectors'
synchronous, CPU-bound `predict()` methods directly (LIME with 400
perturbation samples for text, a ViT/CNN forward pass for images, OpenCV +
EfficientNet for video) without offloading to a thread pool. A single
in-flight scan blocks every other concurrent request on the same worker,
including unrelated users' scans and `/health` checks. For a system meant to
work "during active scrolling" and demoed live to multiple judges at once,
this will manifest as stalls under any concurrency.
**Recommendation:** wrap each `detector.predict(...)` call in
`await run_in_threadpool(detector.predict, ...)` (or make routes `def`
instead of `async def` so Starlette threadpools them automatically). Simple
fix, but changes request-handling behavior under concurrency and should be
verified under load before shipping.

### 3.5 No request-body size limit on JSON endpoints (medium)
Unlike the upload endpoints (capped via `_read_upload_capped`: 10MB image /
100MB video, enforced incrementally), `/scan/text` and `/scan/extension`
place no length constraint on `text` (plain `str`, no `max_length`) or on
`image_urls`. FastAPI buffers the entire body in memory before Pydantic
validation runs, so a caller can POST an arbitrarily large `text` string and
force a large allocation before anything gets rejected — a real availability
risk given this project's documented OOM issues on constrained hosting.
**Recommendation:** add `Field(..., max_length=...)` to `TextScanRequest.text`
(and cap `image_urls` list/string length in `ExtensionScanRequest`), and/or
enforce a global max request-body size at the ASGI/reverse-proxy layer.
Needs a decision on appropriate limits.

---

## 4. Low-severity / judgment calls, still open

- **`training/video/archive.zip` (~1.9GB)** sits directly under
  `training/video/` instead of `data/video/` where the project's data
  convention expects raw data. Gitignored, so not committed, but stray disk
  clutter. Left in place — unclear whether it's still needed without domain
  knowledge of the video pipeline's current state.
- **Extension scope includes Threads** (`threads.net`/`threads.com` in
  `extension/manifest.json`), which CLAUDE.md's north-star doesn't mention
  (only Facebook, TikTok, X named). Not a security issue (still
  least-privilege scoped), but undocumented scope creep. Needs a product-
  scope decision: add Threads to CLAUDE.md's documented v1 scope, or
  remove/defer it from the extension.
- **Unrestricted URL scheme note:** already fixed by the cleanup pass
  (`extension/ui/sidebar.js` now validates `http(s)://` before rendering
  source links) — listed here only for completeness; no action needed.

---

## Already fixed (for reference — do not re-raise)

The cleanup pass following this audit already handled: the stray
`models/image_detector_fp16.pt` artifact, the vestigial
`models/<modality>/.gitkeep` scaffolding, the stale MAX_LENGTH=512 comment in
`training/text/dataset.py`, dead `PhotoSample` fields plus an "Illustrative"
badge on the marketing gallery in `web/src/app/page.tsx`, the
`prism-next-temp` → `prism-web` rename in `web/package.json`, and the
`javascript:`/`data:` URL-scheme gap in `extension/ui/sidebar.js`'s source
links.

A second cleanup pass (2026-08-14, multi-agent hygiene audit — dead
code/files, doc drift, structure) additionally fixed: unused `LABEL_REAL`/
`LABEL_FAKE` constants in `api/modules/text/explainer.py` and `isOpen()`
in `extension/ui/sidebar.js`; the `EfficientNet-B4`
docstring typo in `api/modules/image/detector.py` (actual backbone is B3);
unused `supabase`/`python-dotenv`/`pandas` entries in `api/requirements.txt`;
the stale `MODEL_IMAGE_PATH` var in `api/.env.example`; `video/detector.py`'s
checkpoint-path resolution now matches the `_PROJECT_ROOT` pattern used by
the other two detectors; the undocumented `modules_abstained` field and
missing `supabase/` directory in `README.md`; the extension's unused
`activeTab`/`scripting` permissions, unused `web_accessible_resources`
entries, dead `--prism-ai`/`--prism-human` CSS variables, and vestigial
`short_name` manifest key; the web app's broken `/prism_tab_logo.png`
favicon reference, the resulting orphaned `prism_logo.png` duplicate, and
unused `Menu`/`X` icon imports; stale `python training/<module>/train.py`
usage docs and a wrong resume-checkpoint filename (should be
`-m training.<module>.train`, which is what actually works); the Colab
launcher scripts (`train_image_colab.py`/`train_video_colab.py`) invoking
the trainer with that same broken direct-path form; non-ASCII
arrow/em-dash characters in `training/text/train.py` and
`training/video/train.py`'s "New best F1" print (the same
`UnicodeEncodeError`-on-Windows bug already fixed once in
`training/image/train.py`); a missing shuffle-before-split in
`training/video/train.py` that would have produced a heavily class-skewed
validation set (real-then-fake dataset ordering, sliced unshuffled — same
failure mode `training/image/train.py` already guards against) plus its
now-unused `random_split` import; missing per-batch progress logging in
`training/video/train.py`'s training loop; an unused `import os` in
`training/image/dataset.py`; and dead/inconsistent root config
(`.gcloudignore` for an abandoned GCP deployment path, redundant
`FakeNewsNet` ignore lines, inert `supabase/.branches`+`supabase/.temp`
gitignore rules, `PROBLEMS.md` missing from `.dockerignore` alongside
`loopholes.md`). One new doc/architecture gap was found and added above
rather than silently patched: **2.5**, the "LIME + Anchors" claim when only
LIME is implemented.
