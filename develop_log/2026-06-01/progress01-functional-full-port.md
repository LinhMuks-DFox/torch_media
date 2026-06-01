# Progress 01 — Port the full torchaudio.functional surface (native torch C++)
id: 2026-06-01/progress01
date: 2026-06-01
author: human+ai
status: done
refs: [2026-05-31/progress01, 2026-05-31/progress02, 2026-05-30/progress01]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/_functional.hpp (extend: feature/augment ops)
  - libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp (NEW: lfilter/biquads/effects/vad)
  - libtorchmedia/include/torchmedia/_audio/_functional_beamforming.hpp (NEW: psd/mvdr/rtf/apply_beamforming)
  - libtorchmedia/include/torchmedia/_audio/_functional_alignment.hpp (NEW: forced_align/merge_tokens/token_span)
  - libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp (extend: new option structs)
  - libtorchmedia/include/torchmedia.hpp (aggregate new headers)
  - unit_test/audio/functional/{main.cpp, gen_golden.py} (tests + golden values)
  - .venv (NEW dev-only: torch/torchaudio 2.5.1 authoritative source + golden generator; gitignored)

## Goal
Reimplement the **remaining 53 `torchaudio.functional` v2.5.1 entries (52 functions + the `TokenSpan`
value type)** on native libtorch C++ (`torch::Tensor`) ops, header-only, each shipped with assertion tests
and `.venv` golden cross-checks — completing the full functional surface. The **transform layer is
out of scope here** and follows in a later progress.

## Context / Motivation
Tiers up to progress02 (2026-05-31) ported 8 functional ops. Mux's next milestone is the **complete
functional surface** before building transforms. Authoritative reference is **torchaudio 2.5.1**, now
installed locally in `.venv` (CPU). A 6-agent source-level gap analysis (read torchaudio v2.5.1
`filtering.py` / `functional.py` / `_alignment.py`) produced per-function signature, algorithm,
dependency, torch-native portability, difficulty, and test strategy — those facts are embedded in the
child task files (the `/tmp` analysis artifact is ephemeral; the tasks are the durable record).

### Inventory (torchaudio.functional v2.5.1 `__all__` = 61 functions + `TokenSpan`)
**Already ported (8):** `amplitude_to_DB`, `DB_to_amplitude`, `create_dct`, `melscale_fbanks`
(as `mel_filter_bank`), `spectrogram`, `griffinlim`, `resample`, `convolve`.
(Plus transform-ish helpers already present: `melspectrogram`, `mel_scale`, `mfcc`.)

**To port (53 entries = 52 functions + the `TokenSpan` value type):** lfilter, biquad, filtfilt,
allpass_biquad, lowpass_biquad, highpass_biquad, bandpass_biquad, bandreject_biquad, band_biquad,
equalizer_biquad, bass_biquad, treble_biquad, deemph_biquad, riaa_biquad, contrast, dcshift, gain,
dither, overdrive, phaser, flanger, vad, compute_deltas, linear_fbanks, loudness,
detect_pitch_frequency, sliding_window_cmn, spectral_centroid, inverse_spectrogram, phase_vocoder,
mask_along_axis, mask_along_axis_iid, mu_law_encoding, mu_law_decoding, fftconvolve, add_noise, speed,
preemphasis, deemphasis, pitch_shift, edit_distance, frechet_distance, psd, mvdr_weights_souden,
mvdr_weights_rtf, rtf_evd, rtf_power, apply_beamforming, forced_align, merge_tokens, TokenSpan,
rnnt_loss, apply_codec.

## Decisions

### D1 — Authoritative source & golden values via a local `.venv` (torch/torchaudio 2.5.1)
- Decision: a dev-only `.venv` (gitignored) is the authoritative API source and the golden-value
  generator (`gen_golden.py`). It was created with **linuxbrew `python3.11`** (the system
  `/usr/bin/python3.12` lacks `ensurepip`; torch 2.5.1 has no py3.14 wheels). Install:
  `torch==2.5.1 torchaudio==2.5.1 numpy<2` from the CPU index.
- Why: closed-form + libtorch self-reference cover most cases, but point-wise cross-checks against the
  real torchaudio remove ambiguity for the harder ops (IIR filters, beamforming, VAD trim points).
- Impact: `unit_test/audio/functional/gen_golden.py` grows per task; values are baked into `main.cpp`.

### D2 — File organization mirrors torchaudio's module split
- Decision: new headers `_functional_filtering.hpp` (lfilter + biquads + SoX effects + vad),
  `_functional_beamforming.hpp`, `_functional_alignment.hpp`; feature/augmentation ops extend the
  existing `_functional.hpp`. New option structs go in `_functional_methods_options.hpp`.
  `torchmedia.hpp` aggregates them; everything stays in `torchmedia::audio::functional`.
- Why: keep each header reviewable; group by dependency and theme; preserve header-only ODR rules.
- Alternatives considered: one giant `_functional.hpp` — rejected (already 460 lines; would balloon).

### D3 — `lfilter` is the keystone (sequential IIR recurrence)
- Decision: reimplement torchaudio's pure-torch `_lfilter_core`: FIR numerator via
  `conv1d` (flipped `b`, grouped by channel), normalize by `a0`, then a **sequential C++ time loop**
  for the recursive `a` part; honor `clamp` and the 1D/2D-coeff batching. No custom compiled op.
- Why: it is the only non-vectorizable primitive and `biquad` → all 11 biquad designers, `deemphasis`,
  and `loudness` depend on it. Getting it right and fast first unblocks the largest cluster.

### D4 — Scope-pending items (Mux decides at review)
These four are flagged because feasibility/value is non-trivial; my recommendation noted:
- `forced_align` (L) — **recommend INCLUDE.** Dispatches to a compiled op upstream, but the algorithm
  is a Viterbi-over-CTC-trellis DP that is straightforward as a header-only C++ loop (B=1).
- `vad` (XL) — **recommend INCLUDE (own task).** Feasible header-only (only `fft::rfft`/`hann`/elementwise
  + control flow) but a long stateful state machine (+`_measure` helper); budget accordingly.
- `rnnt_loss` (XL) — **recommend FORWARD-ONLY or DEFER.** Forward cost (alpha lattice in log-space) is
  a feasible C++ loop; the analytic backward + custom autograd `Function` is effectively infeasible
  header-only. Ship a forward-only (no-grad) loss, or defer.
- `apply_codec` (XL) — **recommend DEFER/SKIP.** No tensor math; it is a real sox/ffmpeg codec
  round-trip and is **infeasible** as a pure-torch header-only op (and is deprecated upstream). Only a
  lossless WAV requantization subset is doable via vendored dr_wav; full support belongs behind the
  optional `TORCHMEDIA_WITH_FFMPEG` path.

### D5 — Testing & coverage (unchanged project rule)
- Decision: every ported function ships assertion tests (`TM_CHECK` / `TM_CHECK_TENSOR_CLOSE`) in
  `unit_test/audio/functional/main.cpp`, registered via the existing `audio_test_functional` ctest
  target; target **100% line coverage** of `_functional*.hpp` (vendored excluded). RNG-dependent ops
  (SpecAugment masks) use property/shape tests unless seeds are matched.

### D6 — Dependency-driven execution order
- Tier 0: task01 (`lfilter`/`biquad`/`filtfilt`).
- Tier 1 (after task01): task02 (biquad designers), task12 (`loudness`), task06 (`deemphasis`).
- Independent / any order: task03, task07, task09, task10, task13, task14, task15, task18.
- task08 (`phase_vocoder`) must precede `pitch_shift` (same task).
- Defer/decide: task05 (`vad`) — RESOLVED do last; task16 (`rnnt_loss`) — RESOLVED deferred; task17 (`apply_codec`) — RESOLVED skip.

## Tasks
- [x] [task01 — IIR core: lfilter, biquad, filtfilt](tasks/task01-lfilter-biquad-filtfilt.md) ✅ done
- [x] [task02 — Biquad filter designers (11)](tasks/task02-biquad-designers.md) ✅ done
- [x] [task03 — Simple effects: contrast, dcshift, gain](tasks/task03-effects-contrast-dcshift-gain.md) ✅ done
- [x] [task04 — Modulated-delay effects: overdrive, phaser, flanger](tasks/task04-effects-overdrive-phaser-flanger.md) ✅ done
- [x] [task05 — VAD](tasks/task05-vad.md) ✅ done
- [x] [task06 — Companding & emphasis: mu_law ×2, preemphasis, deemphasis](tasks/task06-companding-emphasis.md) ✅ done
- [x] [task07 — fftconvolve, add_noise, speed](tasks/task07-fftconvolve-addnoise-speed.md) ✅ done
- [x] [task08 — STFT domain: inverse_spectrogram, phase_vocoder, pitch_shift](tasks/task08-stft-phasevocoder-pitchshift.md) ✅ done
- [x] [task09 — Simple feature ops: compute_deltas, linear_fbanks, spectral_centroid](tasks/task09-feature-deltas-linfbank-centroid.md) ✅ done
- [x] [task10 — SpecAugment: mask_along_axis, mask_along_axis_iid](tasks/task10-specaugment-mask.md) ✅ done
- [x] [task11 — Sequential feature ops: sliding_window_cmn, detect_pitch_frequency](tasks/task11-cmn-pitchdetect.md) ✅ done
- [x] [task12 — Loudness (ITU-R BS.1770)](tasks/task12-loudness.md) ✅ done
- [x] [task13 — Metrics: edit_distance, frechet_distance](tasks/task13-metrics-editdistance-frechet.md) ✅ done
- [x] [task14 — Beamforming: psd, mvdr ×2, rtf ×2, apply_beamforming](tasks/task14-beamforming.md) ✅ done
- [x] [task15 — Forced alignment: forced_align, merge_tokens, TokenSpan](tasks/task15-forced-align.md) ✅ done
- [ ] [task16 — rnnt_loss](tasks/task16-rnnt-loss.md) ⏸️ DEFERRED (Mux 2026-06-01)
- [ ] [task17 — apply_codec](tasks/task17-apply-codec.md) ⏭️ SKIPPED (infeasible header-only; Mux 2026-06-01)
- [x] [task18 — dither (TPDF/RPDF/GPDF + noise shaping)](tasks/task18-dither.md) ✅ done

## Issues / Gotchas
- `lfilter`: coeffs ordered `[a0,a1,...]`, both flipped + `/a0` normalized; the time loop is the only
  non-vectorizable piece; `bass_biquad` pre-divides by a0 but `treble_biquad`/most others do not.
- Beamforming needs **complex** `torch::linalg::solve` / `eigh`; eigenvectors are phase-ambiguous
  (test phase-invariant quantities).
- SpecAugment masks are RNG-dependent (prefer property tests).
- `vad` integer bin/index math (`dft_len_ws`, spectrum/cepstrum ranges) must match exactly or the trim
  index diverges.
- `.venv` is dev-only and gitignored; CI/coverage must not depend on it at build time (golden values
  are baked into `main.cpp`).

## Open / TODO (carry-over)
- The **transform layer** (`torchaudio.transforms`) — next progress, after functional is complete.
- Earlier carry-overs: `mel_filter_bank` vectorization; vision Tier-2 transforms; plot colormap/mel
  convenience plots + gnuplot 6.0 warning suppression.

## Agent log
- 2026-06-01 [ai] Created `.venv` (linuxbrew py3.11, torch/torchaudio 2.5.1+cpu) as authoritative
  source + golden generator. Ran a 6-agent gap analysis over torchaudio v2.5.1 source; produced
  per-function signature/algorithm/portability/difficulty/test-strategy for all 53 unported entries.
  Authored this progress + 18 task files (task18 `dither` added after a completeness re-check against
  `torchaudio.functional.__all__` caught it as the one entry missed by the initial grouping; coverage
  is now 53/53). **No implementation yet** — Mux explicitly deferred coding
  ("先别动手实现，晚一点我让你继续做"). Next: on Mux's go, start task01 (lfilter keystone) and resolve
  the D4 scope-pending decisions.
- 2026-06-01 [ai] Mux resolved D4 scope: vad -> do LAST; rnnt_loss -> DEFER (not this milestone);
  apply_codec -> SKIP (infeasible header-only, deprecated upstream). task16/task17 marked accordingly.
  Implemented so far (green, golden-checked): task01/02/03/06/07/09/13 (29 functions). ctest 4/4.
- 2026-06-01 [ai] COMPLETE (per Mux's D4 scope). Implemented 50 functions + the `token_span` type across
  tasks 01-15 + 18 — all green: `audio_test_functional` 189/189, `ctest` 4/4. New headers:
  `_functional_filtering.hpp` (lfilter/biquads/effects/dither/vad), `_functional_beamforming.hpp`,
  `_functional_alignment.hpp`; feature/augment ops extend `_functional.hpp`. Golden values for every op
  baked from the `.venv` (torchaudio 2.5.1) via `gen_golden.py`. Out of scope per Mux: `rnnt_loss`
  (task16, DEFERRED) and `apply_codec` (task17, SKIPPED — infeasible header-only). Coverage of
  `torchaudio.functional.__all__`: 51/53 entries implemented, 2 intentionally out-of-scope.
  Not yet committed — awaiting Mux's review. Next milestone: the transforms layer (new progress).
