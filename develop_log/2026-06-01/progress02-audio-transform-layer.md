# Progress 02 — Audio transform layer (torchaudio.transforms → torchmedia::audio::transform)
id: 2026-06-01/progress02
date: 2026-06-01
author: human+ai
status: done
refs: [2026-06-01/progress01, 2026-05-31/progress01, 2026-05-31/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/_transform.hpp (becomes the aggregator)
  - libtorchmedia/include/torchmedia/_audio/_transform_spectral.hpp (NEW)
  - libtorchmedia/include/torchmedia/_audio/_transform_feature.hpp (NEW)
  - libtorchmedia/include/torchmedia/_audio/_transform_augment.hpp (NEW)
  - libtorchmedia/include/torchmedia/_audio/_transform_effects.hpp (NEW)
  - libtorchmedia/include/torchmedia/_audio/_transform_beamform.hpp (NEW)
  - libtorchmedia/include/torchmedia/_audio/_functional.hpp (factor resample kernel build/apply — see G1)
  - unit_test/audio/transform/{CMakeLists.txt, main.cpp, gen_golden.py} (NEW test target)
  - unit_test/audio/CMakeLists.txt (wire in the new subdir)

## Goal
Reimplement the **35 in-scope `torchaudio.transforms` v2.5.1 classes** (36 total minus `RNNTLoss`)
as header-only, native-libtorch **stateful classes** in `torchmedia::audio::transform`, each
delegating to the now-complete `torchmedia::audio::functional` surface where possible and shipping
assertion tests with `.venv` (torchaudio 2.5.1) golden cross-checks. This introduces the library's
**first classes**; functional was free-functions-only. Vision transforms are out of scope (separate
future milestone).

## Context / Motivation
progress01 (2026-06-01) completed the functional surface (51/53). The transform layer is the next
milestone. A source-level read of torchaudio 2.5.1 (`transforms/_transforms.py`,
`transforms/_multi_channel.py`) established **what the transform layer adds beyond wrapping
functional** — which is the rationale for making it class-based:

1. **Cached precomputed state** (the primary reason classes exist): the STFT window, the mel /
   linear filterbank matrix, the DCT matrix, the sinc resample kernel, the phase-advance vector —
   each built once in `__init__` and reused every call (functional recomputes per call).
2. **Hand-written ops with no functional counterpart**: `MelScale`/`InverseMelScale`/`MFCC`/`LFCC`
   apply the fbank/DCT *in forward* by `matmul` (the matrix is built by `F.*` in `__init__` but
   applied by plain torch); `InverseMelScale` solves `torch.linalg.lstsq`+`relu`; `Fade` synthesizes
   fade curves; `Vol` dispatches gain modes.
3. **Composition** of sub-transforms (MelSpectrogram = Spectrogram+MelScale; MFCC =
   MelSpectrogram+AmplitudeToDB+DCT; LFCC = Spectrogram+AmplitudeToDB+fbank+DCT; Speed = Resample;
   SpeedPerturbation = ModuleList[Speed]; MVDR = PSD).
4. **Runtime/streaming state**: `PitchShift` lazy kernel; `SpeedPerturbation` per-call RNG;
   `MVDR` (online) recursive cross-call PSD accumulation (4 mutable buffers).
5. **Validation / default-derivation / batch shape pack-unpack.**

### Inventory (torchaudio.transforms v2.5.1 = 36 classes)
**In scope (35):** Spectrogram, InverseSpectrogram, GriffinLim, SpectralCentroid, AmplitudeToDB,
MelScale, InverseMelScale, MelSpectrogram, MFCC, LFCC, Resample, Speed, SpeedPerturbation,
PitchShift, TimeStretch, FrequencyMasking, TimeMasking, SpecAugment, Fade, MuLawEncoding,
MuLawDecoding, Preemphasis, Deemphasis, ComputeDeltas, SlidingWindowCmn, Loudness, Vad, Vol,
Convolve, FFTConvolve, AddNoise, PSD, SoudenMVDR, RTFMVDR, MVDR. (`_AxisMasking` is an internal
base for the two Masking classes.)

**Out of scope (1):** `RNNTLoss` — its functional `rnnt_loss` is DEFERRED (progress01 task16).

## Decisions

### D1 — Stateful classes + `operator()` (first classes in the library)
- Decision: each transform is a class in `torchmedia::audio::transform`. The constructor takes the
  config and precomputes/caches buffers; `tensor_t operator()(const tensor_t&) const` runs it, with
  a `forward(...)` alias for torchaudio parity. Stateless-after-construction except where the source
  demands runtime state (D4).
- Why: the source analysis shows the layer's value is cached state + composition + a little
  hand-written logic — none of which a free-function alias delivers.
- Impact: introduces classes; option structs stay value types.

### D2 — Naming: PascalCase classes
- Decision: PascalCase class names 1:1 with torchaudio (`transform::Spectrogram`, `MelSpectrogram`,
  `MFCC`, `MVDR`). Option structs stay `snake_case` `xxx_option` per project convention.
- Why: 1:1 with the Python API eases porting and naturally distinguishes a class
  (`transform::Spectrogram`) from a function (`functional::spectrogram`).

### D3 — Scope: all 35 in one progress, RNNTLoss deferred
- Decision: cover all 35 in-scope classes in this single progress, tiered into tasks 01–07.
  `RNNTLoss` deferred (follows its functional dependency).

### D4 — MVDR online: full faithful port
- Decision: port `MVDR` completely, **including** the online recursive PSD accumulation
  (higuchi2017online): 4 mutable accumulator members (`psd_s`, `psd_n`, `mask_sum_s`, `mask_sum_n`)
  blended across calls, with an explicit `bool initialized_` first-call sentinel (replacing
  torchaudio's `ndim==1` trick).
- Why: Mux wants a faithful port; online mode is the distinguishing feature of the class.
- Impact: `MVDR::operator()` is **non-const** (mutates accumulators); see G2.

### D5 — Independent option struct per transform
- Decision: every transform defines its **own** `xxx_option` struct in the `transform` namespace
  (no reuse of functional's option structs), `snake_case` with fluent setters returning `*this`;
  fields may duplicate functional's. String options become C++ enums (`convolve_mode` already
  exists in functional; add `fade_shape`, `gain_type`, `lstsq_driver`, `mvdr_solution`).
- Why: decouples the transform API from functional's option evolution; enums make invalid states
  unrepresentable (replacing torchaudio's runtime string `ValueError`s).

### D6 — File split mirrors functional
- Decision: `_transform_spectral.hpp` (Spectrogram, InverseSpectrogram, GriffinLim,
  SpectralCentroid, MelScale, InverseMelScale, MelSpectrogram), `_transform_feature.hpp`
  (AmplitudeToDB, MFCC, LFCC, ComputeDeltas, SlidingWindowCmn, SpectralCentroid grouping per
  implementer), `_transform_augment.hpp` (Masking ×2 + base, SpecAugment, Fade, Resample/Speed/
  PitchShift/TimeStretch time-domain — or a `_transform_time.hpp`, implementer's call),
  `_transform_effects.hpp` (companding/emphasis/Vol/Convolve/FFTConvolve/AddNoise/Loudness/Vad),
  `_transform_beamform.hpp` (PSD, MVDR ×3). `_transform.hpp` aggregates them. The exact
  header-to-class mapping is the implementer's call provided each header is reviewable and
  include-ordered (composition deps first). Option structs live alongside each header or in a
  shared `_transform_options.hpp`.
- Why: keep each header reviewable; mirror the functional split precedent (progress01 D2).

### D7 — Testing & coverage (unchanged project rule)
- Decision: new `unit_test/audio/transform/` target `audio_test_transform`, mirroring
  `unit_test/audio/functional/`. Golden values generated from `.venv` `torchaudio.transforms`
  2.5.1 via `gen_golden.py`, baked into `main.cpp`, compared with `TM_CHECK_TENSOR_CLOSE`. RNG
  classes (`*Masking`, `SpecAugment`, `SpeedPerturbation`) use property/shape tests. Target 100%
  line coverage of `_transform*.hpp` (vendored excluded). `ctest` must be green before any task is
  done.

### D8 — PitchShift kernel built eagerly
- Decision: build the sinc resample kernel **eagerly in the constructor** (we know dtype/device or
  take them as options); do not replicate `LazyModuleMixin`.
- Why: header-only C++ has no clean lazy-module analogue; eager construction is simpler and the
  cache is the point.

## Tasks
Task numbers continue the 2026-06-01 sequence (progress01 used task01–18; task IDs are date-scoped
per `develop_log/README.md`).
- [x] [task19 — Transform pattern + window-buffer spectral primitives](tasks/task19-transform-pattern-spectral-primitives.md) ✅ done
- [x] [task20 — Mel & cepstral (matmul-projection + compositors)](tasks/task20-mel-cepstral.md) ✅ done
- [x] [task21 — Resample & time-domain](tasks/task21-resample-time-domain.md) ✅ done
- [x] [task22 — Augmentation (masking, SpecAugment, Fade)](tasks/task22-augmentation.md) ✅ done
- [x] [task23 — Companding / emphasis / feature thin wrappers + Vol](tasks/task23-thin-wrappers.md) ✅ done
- [x] [task24 — Convolution & noise](tasks/task24-convolution-noise.md) ✅ done
- [x] [task25 — Beamforming (PSD, MVDR incl. online)](tasks/task25-beamforming.md) ✅ done

Dependency: only **task20 depends on task19** (composition needs `Spectrogram`/`AmplitudeToDB`).
task21–25 are independent and may proceed in any order / in parallel.

## Issues / Gotchas
- **G1 — Resample/PitchShift kernel caching.** `functional::resample` is monolithic (rebuilds the
  sinc kernel each call). To actually cache it (the whole point of `transform::Resample`), factor
  functional into `functional::detail::sinc_resample_kernel(...)` +
  `functional::detail::apply_sinc_resample_kernel(...)`, have `functional::resample` call both, and
  have `transform::Resample`/`PitchShift` cache the kernel and call only the apply helper. This is
  the one functional edit this milestone requires; keep `functional::resample`'s public behavior
  byte-identical (golden values from progress01 must still pass).
- **G2 — MVDR online state.** The 4 accumulators mutate across calls, so `MVDR::operator()` cannot
  be `const`. First-call init via an explicit `bool initialized_`. Beamforming uses complex
  `linalg::solve`/`eigh`; eigenvectors are phase-ambiguous — test phase-invariant quantities (carry
  over from progress01 beamforming gotcha).
- **G3 — Compose, don't re-wrap.** `MelSpectrogram`/`MFCC`/`LFCC` must compose cached
  sub-transforms (`Spectrogram`+`MelScale`, etc.) so the window/fbank/DCT are actually cached. Do
  NOT call `functional::melspectrogram`/`functional::mfcc` (which recompute the fbank/DCT each
  call) — that would defeat the layer's purpose.
- **G4 — `InverseMelScale` has no functional counterpart.** Implement the lstsq solve in the
  transform: `relu(linalg::lstsq(fb.transpose(-1,-2)[None], mel, driver).solution)`, with a
  `lstsq_driver` enum {gels,gelsy,gelsd,gelss}; validate input `n_mels` and do the batch view
  pack/unpack `view(-1, n_mels, time)` ↔ restore.
- **G5 — RNG/determinism.** Mask placement RNG lives inside `functional::mask_along_axis(_iid)`;
  `SpeedPerturbation` draws a factor per call. Tests are property/shape-based. Decide a seeding
  policy (e.g. accept an optional generator/seed) when wiring the masking classes.
- **G6 — Drop dead args.** Do not port the deprecated `return_complex` / `onesided` warnings.
- **G7 — `MVDR` dtype promotion.** Input `cfloat` is promoted to `cdouble` internally and cast
  back to the original dtype on output; mirror this.

## Open / TODO (carry-over)
- Earlier carry-overs (from progress01): `mel_filter_bank` vectorization; plot colormap/mel
  convenience plots; gnuplot 6.0 warning suppression.
- Vision transforms (`_vision/_transform`) — separate future milestone.

## Agent log
- 2026-06-01 [ai] Ran a 6-agent source-level read of torchaudio 2.5.1 transforms
  (`_transforms.py`, `_multi_channel.py`) producing per-class facts (functional calls,
  `__init__` precomputed buffers, new logic, composition, runtime state, validation). Confirmed the
  4 key decisions with Mux: D1 stateful classes + `operator()`, D2 PascalCase, D3 all 35 in one
  progress (RNNTLoss deferred), D4 MVDR online full port, D5 independent option structs.
  Authored this progress + 7 task files (task19–task25, continuing the date's task sequence).
- 2026-06-01 [ai] task19 DONE (keystone). NEW `_transform_spectral.hpp` (Spectrogram,
  InverseSpectrogram, GriffinLim, SpectralCentroid) establishing the class idiom; `_transform.hpp`
  now aggregates it. NEW `audio_test_transform` ctest target. Each class caches its Hann window and
  delegates to the golden-verified `functional::` op; tests assert delegation-equivalence (exact) +
  baked torchaudio.transforms golden. `audio_test_transform` 29/29, `ctest` 5/5, **100% coverage of
  `_transform_spectral.hpp`**. Not yet committed — awaiting Mux's review.
- 2026-06-01 [ai] task20 DONE (mel/cepstral). NEW `_transform_feature.hpp`: AmplitudeToDB, MelScale,
  InverseMelScale (lstsq, `lstsq_driver` enum), MelSpectrogram, MFCC, LFCC. Key fix: `MelScale`
  projects via `matmul(_fb, spec)` to preserve input rank (functional::mel_scale collapses leading
  dims into one batch axis; caught by the torchaudio golden). MelSpectrogram/MFCC/LFCC compose cached
  sub-transforms (G3). `audio_test_transform` 57/57, `ctest` 5/5, **100% line/region/function/branch
  coverage** of both transform headers. Not committed — awaiting review. Next: task21–25 (all
  independent; task21 includes the G1 functional resample-kernel split).
- 2026-06-01 [ai] BUGFIX (Mux-approved) in `functional::mel_scale`: it collapsed all leading dims
  into a single batch axis, so 1D/2D input gained a spurious `(1, ...)` axis and >3D input lost its
  batch/channel structure (torchaudio preserves rank). Replaced the reshape/transpose dance with a
  rank-preserving `matmul(fb, spec)` -> `(..., n_mels, time)`. `transform::MelScale` reverted to
  delegate to the fixed `functional::mel_scale` (single implementation; dropped the test `squeeze(0)`
  workarounds). Existing functional tests use a 2D `(1, N)` waveform (3D spectrogram) so their output
  rank is unchanged: `audio_test_functional` 189/189, `audio_test_transform` 57/57, `ctest` 5/5 all
  green; transform headers stay 100%-covered. Side finding (out of scope, pre-existing): `_functional.hpp`
  is NOT actually at 100% line coverage — ~34 error/branch lines (convolve pad modes, add_noise
  length-mask, mask_along_axis validation, inverse_spectrogram norm/pad branches, sliding_window_cmn)
  are untested; flagged for a later coverage-hardening pass.
- 2026-06-01 [ai] COVERAGE HARDENING (Mux-approved) of `_functional.hpp`: added `test_coverage_gaps`
  to `audio_test_functional` exercising the previously-untested paths — `mu_law_encoding` integer
  cast, `compute_deltas` constant/reflect/circular pad modes, `add_noise` length-mask,
  `frechet_distance` both shape-validation raises, `mask_along_axis` frequency-axis unsqueeze,
  `mask_along_axis_iid` ndim/axis/p raises + mp<1 early return, `inverse_spectrogram` window-norm and
  pad-trim branches, `sliding_window_cmn` window-clamp + norm_vars (incl. single-frame zero-fill).
  `_functional.hpp` now **100% line + 100% function coverage** (814/814 lines; remaining ~13 partial
  branches are `&&`/`||`/ternary short-circuits — line coverage is the project's documented bar).
  `audio_test_functional` 206/206, `ctest` 5/5 green. Not committed — awaiting review.
- 2026-06-01 [ai] task21 DONE (resample & time-domain). G1: split `functional::resample` into
  `_sinc_resample_kernel` + `_apply_sinc_resample_kernel` (behavior byte-identical; resample golden
  still green). NEW `_transform_time.hpp`: Resample (cached kernel + identity short-circuit), Speed
  (compose Resample + out-lengths), SpeedPerturbation (global-RNG factor pick), PitchShift (eager
  cached kernel), TimeStretch (cached phase_advance + rate-null raise). Tests = delegation-equivalence
  to functional + RNG property/repro + raise. `audio_test_transform` 82/82, `ctest` 5/5,
  `_transform_time.hpp` 100% lines, `_functional.hpp` still 100% lines. Not committed — awaiting review.
- 2026-06-01 [ai] task22 DONE (augmentation). NEW `_transform_augment.hpp`: AxisMasking base +
  FrequencyMasking/TimeMasking, SpecAugment (mask loops, no time-warp), Fade (5-shape curve synthesis,
  enum). Masking tested by seeded delegation-equivalence; Fade by torchaudio golden (5 shapes).
  `audio_test_transform` 100/100, `ctest` 5/5, `_transform_augment.hpp` 100% lines. Not committed —
  awaiting review.
- 2026-06-01 [ai] task23 DONE (thin wrappers + Vol). NEW `_transform_effects.hpp`: MuLawEncoding/
  Decoding, Preemphasis, Deemphasis, ComputeDeltas, SlidingWindowCmn, Loudness, Vad (forwards 17
  params), Vol (gain-mode enum + clamp + negative-gain raise). Tested by delegation-equivalence +
  Vol mode/clamp/raise. `audio_test_transform` 123/123, `ctest` 5/5, `_transform_effects.hpp` 100%
  lines. Not committed — awaiting review. Next: task24 (convolution & noise).
- 2026-06-01 [ai] task24 DONE (convolution & noise). Added Convolve/FFTConvolve (reuse
  `functional::convolve_mode` enum) + AddNoise to `_transform_effects.hpp`; delegation-equivalence
  tests over all 3 modes + AddNoise lengths variants. `audio_test_transform` 133/133, `ctest` 5/5,
  all 5 transform headers 100% line+function coverage. Not committed — awaiting review. Next: task25
  (beamforming incl. MVDR online — the last task).
- 2026-06-01 [ai] task25 DONE (beamforming) — MILESTONE COMPLETE. NEW `_transform_beamform.hpp`:
  PSD, SoudenMVDR, RTFMVDR, MVDR (3 solutions + cfloat→cdouble + full online recursive PSD with
  mutable members + first-call sentinel; non-const operator()). Exact functional-replication tests.
  `audio_test_transform` 151/151, `ctest` 5/5, `_transform_beamform.hpp` 100% all dimensions.
- 2026-06-01 [ai] **progress02 COMPLETE.** All 35 in-scope `torchaudio.transforms` classes implemented
  across 6 headers (`_transform_{spectral,feature,time,augment,effects,beamform}.hpp`, aggregated by
  `_transform.hpp`), establishing the library's first stateful class layer (D1/D2). RNNTLoss remains
  out of scope (functional dependency deferred). Side work: fixed the `functional::mel_scale` rank bug
  and hardened `_functional.hpp` to 100% line coverage. Final state: `audio_test_transform` 151/151,
  `audio_test_functional` 206/206, `ctest` 5/5; **every `_transform*.hpp` and `_functional.hpp` at
  100% line + function coverage**; clang-format clean. Not committed — awaiting Mux's review of the
  full milestone. Carry-over: the few partial-branch (`&&`/`||`/switch-default) gaps in time/augment/
  effects headers (line coverage is the project bar); RNNTLoss + vision transforms for a future
  milestone.
