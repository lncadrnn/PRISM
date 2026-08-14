# PRISM — loopholes / conceptual gaps

Honest critical read of the project, prompted by the question: "disinformation
and AI/non-AI detection are two different things — why conjoin them, or is it
alright?" Short answer: **you're right to be suspicious.** Below is what's
actually happening in the code (not just the docs), why it's a real problem,
and what to do about it. Written for a thesis defense, so it names the sharp
edges directly rather than softening them.

---

## 1. The core issue: "credibility" and "authenticity" are different axes, but they get averaged into one number

- **Credibility** (text module): is the *claim* true? A fact-value judgment.
- **Authenticity** (image/video module): is this *pixel content* AI-generated? A provenance judgment.

These are orthogonal. A real, unedited photo can accompany a completely false
claim (the dominant real-world disinformation pattern: authentic media,
false/misleading context). An AI-generated image can illustrate a true,
well-sourced claim (stock-art headers, satire clearly labeled as such, VFX).
Knowing one tells you almost nothing about the other.


Your own code already says this, in `extension/lib/verdict.js` (lines 34-35):

> "These are orthogonal: a real photo can carry a false claim, and an AI
> image can illustrate a true one."

But `api/fusion/__init__.py` averages them anyway:

```python
_WEIGHTS: dict[str, float] = {"text": 0.40, "image": 0.35, "video": 0.25}
fake_score = sum(v.confidence * _WEIGHTS[k] for k, v in present.items()) / total_weight
```

**Worked example with your actual weights** (video absent, so text/image
renormalize to 0.533/0.467): a real photo (image confidence, i.e. P(AI) = 0.02)
attached to a moderately-false claim (text confidence, i.e. P(fake) = 0.6):

```
fake_score = 0.6 × 0.533 + 0.02 × 0.467 = 0.320 + 0.009 = 0.329 → "real"
```

A genuinely misleading post — a false claim riding on an authentic photo —
gets **diluted down to "LIKELY CREDIBLE"** by the extension's own headline
logic, purely because the photo happens to be a real camera photo. The
photo's authenticity has zero bearing on whether the claim is true, but the
fusion math treats it as corroborating evidence anyway.

Run it the other way — a true, well-sourced claim (text P(fake)=0.3) with an
AI-generated header illustration (image P(fake)=0.9):

```
fake_score = 0.3 × 0.533 + 0.9 × 0.467 = 0.160 + 0.420 = 0.580 → "fake"
```

A **true** post gets branded **"FAKE & NON-CREDIBLE"** because of a decorative
AI image that has nothing to do with the claim's truth.

This isn't a hypothetical edge case — it's the direct, structural consequence
of averaging two measurements of different things.

## 1a. A second averaging error, same shape: the text module's own architecture

Not a fusion bug this time, but the same underlying pattern — a plan that
distinguishes two things carefully, and an implementation that quietly
collapses them. CLAUDE.md specifies the text module as a **project fine-tune**
of DistilBERT-Tagalog on Vera Files/AFP/FakeNewsNet data, chosen specifically
for its small size (~40% of teacher) to hit real-time latency on consumer
hardware. What actually runs in production (`api/modules/text/model.py`
`MODEL_ID`, `api/modules/text/detector.py`) is the unmodified third-party hub
checkpoint `iceman2434/xlm-roberta-base-fake-news-detection-tl` — an
XLM-RoBERTa-**base** model, ~3x larger than a distilled model, never fine-tuned
by this project on any Taglish/Vera-Files data. `models/text/` never held a
real checkpoint. `training/text/train.py` even contains a warning
acknowledging a checkpoint trained on a different backbone isn't loadable by
the current inference `TextDetector` — i.e. the divergence is known, not
accidental. **Recommendation:** treat this the same way as point 1 —either
update CLAUDE.md to state the real production backbone and drop the
DistilBERT-Tagalog claim, or actually run the fine-tune/distillation step
before a defense, and measure latency against the <2s budget once you do,
since XLM-R-base is a real risk to that number on i5/Ryzen5-class hardware.

## 2. The fix already half-exists, but isn't finished

The extension's two-axis badge design (credibility segment + authenticity
segment, shown separately) is the *right* instinct — it's explicitly built to
avoid exactly this conflation. But `deriveAxes()` in `verdict.js` still
computes an `overall` field straight from the fused `/scan/extension` score
(`"FAKE & NON-CREDIBLE"` / `"LIKELY CREDIBLE"` / `"INCONCLUSIVE"`) and
presumably surfaces it as a headline somewhere in the UI (check `badge.js`/
`sidebar.js` for where `overall` gets rendered). So the product currently
ships **both** the correct two-axis breakdown **and** the conflated single
verdict, side by side, contradicting itself.

**Recommendation:** either drop the fused `overall` headline entirely and
lead with the two axes (matches what your own code comment already argues
for), or if you keep a single headline for simplicity, derive it from
whichever axis is worse (e.g. "non-credible" wins over "AI-generated" as the
more actionable warning) rather than a weighted average of confidences that
mean different things.

## 3. The video module isn't actually wired into the live product

Traced through `api/main.py`:

- `/scan` (multipart upload, used by the web app for manual submission) —
  **correctly** runs an uploaded video file through `video_detector.predict`.
- `/scan/extension` (the **only** endpoint the browser extension calls, per
  `extension/background/service-worker.js`) — `verdicts["video"]` is
  initialized to `None` and **never assigned**. For a video post, the
  extension (`extractPosterUrl` in `scanner.js`) sends the poster/thumbnail
  as an `image_url`, which gets run through the **image** detector, not the
  video one.

So for the flagship real-time scanning use case (per the North-star
sentence — "scan a social media post in real time"), TikTok/Reels/X-video
posts never touch the video module's spatial+temporal forensic pipeline you
built in `training/video/` at all. The badge that appears on a video post is
running still-image AI-detection on one frame, then labeling that as the
"authenticity" signal for a *video*. This isn't disclosed anywhere in the UI —
a user has no way to know "AI" on a video badge actually means "the poster
frame looked AI-generated," not "we analyzed this video for deepfake
artifacts."

**This is worth fixing or explicitly disclosing before a defense** — a
panelist who reads `training/video/dataset.py`'s frame-extraction pipeline
and then opens the extension in a browser will notice the mismatch.

## 4. Even where it works as designed, the *target* is narrower than "disinformation"

The image/video modules detect **synthetic generation** (GAN/diffusion
artifacts) — not manipulation of authentic media (cropping, recoloring,
old-photo-passed-off-as-current, miscaptioning), which is the more common
disinformation technique in practice, especially in Filipino social media
political content. A system that can perfectly detect "is this AI-generated"
still misses most real-world visual disinformation, because most of it isn't
AI-generated at all — it's real media with a fabricated context. "Disinformation
detection system" oversells what the image/video modules can actually catch;
"AI/synthetic-media + text-claim forensics" is the more honest description of
what's implemented.

## 5. Weight tuning can't fix a category error

`training/README.md` and the fusion weights comment ("research-paper
weights") suggest the 0.40/0.35/0.25 split is calibratable. It is — but
calibration only changes *how much* of the wrong kind of evidence gets
mixed in, not *whether* it should be mixed in at all. No weight setting
makes "is this AI-generated" a valid proxy for "is this claim true."

## 6. Compounding domain-mismatch gaps (already partially self-acknowledged)

- Text module: pretrained on FakeNewsNet, which is **English, title-only**
  political/gossip news (per `training/README.md`'s own warning) — the actual
  target domain is Taglish social media disinformation. The in-domain
  Filipino fact-checked dataset (Vera Files / AFP Philippines) still needs to
  be sourced and fine-tuned on.
- Image module: trained on CIFAKE (32×32 CIFAR-derived) + GRAVEX-200K —
  generic real-vs-AI-image datasets, not Filipino social media imagery, and
  not annotated for deceptive use at all. The model learns "AI-generated
  or not" in a vacuum, disconnected from any disinformation context.

Neither of these is a new discovery — both are already flagged in your own
docs — but they compound points 1-4: the modules are trained to answer
generic questions (is this claim false in general / is this image
AI-generated in general), not the specific question the thesis is framed
around (is this Filipino social media post disinformation).

## 7. UX/ethical risk of the single verdict

Per `CLAUDE.md`'s own rule: PRISM is a decision-support tool, not a
moderator, and should never imply an authoritative fact-check. But a bold
"FAKE & NON-CREDIBLE" headline — especially one partly driven by an
orthogonal authenticity signal per point 1 — risks being read by users as
exactly that authoritative judgment. This is worth addressing in the UX
even independent of the fusion-math fix.

---

## Bottom line

Conjoining them into **one fused number** the way `api/fusion/__init__.py`
currently does: **not alright** — the worked examples above show it produces
wrong verdicts in both directions (masking real disinformation behind an
authentic photo, and branding true content "fake" because of an AI-generated
illustration). Presenting them as **two separate, clearly-labeled axes** (which
the extension already does alongside the fused number) is the defensible
design — you were already halfway to the right answer before this
conversation. The remaining work is: (a) stop computing/displaying the fused
single verdict, or at least stop calling it "credibility," and (b) either wire
the video module into the extension's real-time path for real, or explicitly
scope the thesis to say the passive-scanning flow currently uses
image-based authenticity as a proxy for video.

---

## 8. Loopholes surfaced by the full-repo audit + the Render deployment saga

A separate full-repo audit (frontend/backend/QA) plus the Render.com
deployment attempt (`docs/render-deployment-log.md`) turned up more gaps in
the same spirit as points 1-7: places where the plan says one thing and the
shipped code does another, or where a real security/reliability hole exists
underneath a feature that otherwise "works." Concrete fixes below, not
repeated here in full — see `PROBLEMS.md` for the complete unresolved-findings
list this section summarizes.

**Security holes in `api/main.py` that undercut the "decision-support, not
authoritative" posture (point 7 above):** the SSRF guard on `/scan/extension`
(`_is_public_http_url`) checks the *original* hostname's resolved IP but then
fetches with `follow_redirects=True` without re-checking each redirect hop —
a compromised or malicious public URL can 302 the fetcher straight at
`169.254.169.254` or an internal service. Separately, the rate limiter
(`_client_ip`) trusts a client-supplied `X-Forwarded-For` unconditionally, so
anyone can spoof a fresh "IP" per request and both bypass the 20 req/min cap
and leak memory into `_request_log` forever. **Recommendation:** re-validate
every redirect hop against the same public-IP allowlist (or pin the
connection to the already-checked IP), and only trust `X-Forwarded-For` from
a known trusted proxy, taking the right-most hop rather than the spoofable
left-most one. Neither fix is architectural — both are containable to
`api/main.py` — but both are "ship a live system to strangers and get burned"
risk given this is a real endpoint the extension calls on every scanned post.

**No regression net anywhere:** zero tests, no CI, and `.gitignore` actively
excludes `api/smoke_*.py`/`api/test_*.py`, so even a developer's own local
sanity scripts never become a shared regression guard. Given how much of this
document is "the code quietly does something different from the plan,"
that's not a coincidence — a test asserting `fuse()`'s behavior on a
missing-modality post, or a test asserting `/scan/extension`'s SSRF guard
rejects a redirect to a private IP, would have caught several of these gaps
mechanically instead of via a manual read-through. **Recommendation:** a
minimal pytest suite (fusion edge cases, `/scan*` route smoke tests, SSRF/
rate-limit unit tests with mocked DNS) wired into a basic GitHub Actions
workflow, before the next round of changes rather than after.

**Two modules with no trained checkpoint behind the confident-sounding
plan language:** `models/text/`, `models/image/`, `models/video/` were all
just `.gitkeep` scaffolding (the image one is populated at runtime; text and
video never got a real fine-tuned checkpoint at all). The video module's
`forensics.py` docstrings assert its ELA/optical-flow thresholds are
"empirically calibrated on FaceForensics++," but no calibration script,
notebook, or dataset backing that claim exists anywhere in the repo — the
numbers are uncalibrated guesses wearing a validated-sounding comment.
**Recommendation:** either do the calibration/fine-tuning work and cite it,
or strip the claim down to "hand-tuned starting thresholds, not yet
validated against a labeled set" — an unverifiable claim of rigor is worse
for a thesis defense than an honest gap.

**Evaluation harness (build step 9) doesn't exist yet**, which means every
number in points 1-7 above (and the Bento section's F1/latency figures on the
public web page) is argued from code-reading and worked examples, not from a
confusion matrix run against a held-out set. This is the single biggest gap
standing between "the architecture is defensible" and "the architecture is
proven" — until it exists, none of the ≥90% F1 / <2s latency / ≥68 SUS
targets in CLAUDE.md are actually measured claims.

**The Render free-tier wall is a capacity loophole, not a code loophole:**
importing `torch`+`torchvision`+`transformers`+`opencv` alone consumes ~85%
of Render's 512MB free-tier ceiling before any inference runs, and one
forward pass (let alone one with GradCAM) blows well past it. No further
code-level trick (fp16 was tried and reverted after producing `NaN` logits on
real images; meta-device + `assign=True` loading was the one fix that
actually helped, saving ~750MB) closes a gap this architectural. The three
real options are: (1) run inference on your own machine behind a free tunnel
(Cloudflare Tunnel/ngrok — zero cost, but requires the machine to be on), (2)
a small paid tier (~$7/mo Render Starter, ~2GB RAM — directly solves it but
reopens the "no payments" constraint), or (3) a genuinely lighter runtime
(quantized ONNX Runtime instead of full PyTorch — plausible but an
unattempted rewrite with no fit guarantee). See
`docs/render-deployment-log.md` for the full profiling table and commit-by-
commit story, and `PROBLEMS.md` for this written up as an open decision the
project owner needs to make before the next deploy attempt.
