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
