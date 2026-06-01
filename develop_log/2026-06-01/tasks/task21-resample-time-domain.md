# Task 21 — Resample & time-domain (Resample, Speed, SpeedPerturbation, PitchShift, TimeStretch)
id: 2026-06-01/task21
parent: 2026-06-01/progress02
status: done              # active | blocked | done
owner: code_agent

## Result
Done 2026-06-01. **G1 functional refactor**: split `functional::resample` into `_sinc_resample_kernel`
(build) + `_apply_sinc_resample_kernel` (pad/strided-conv/crop), with `resample` calling both — public
behavior byte-identical (`audio_test_functional` 206/206, resample golden unchanged). NEW
`_transform_time.hpp` (aggregated by `_transform.hpp`): `Resample` (caches the sinc kernel via the G1
helper; identity short-circuit), `Speed` (composes Resample + out-length tracking), `SpeedPerturbation`
(vector<Speed> + per-call global-RNG factor pick — matches torchaudio; reproducible via
`torch::manual_seed`), `PitchShift` (stft→phase_vocoder→istft via `functional::_stretch_waveform`, then
the cached eager resample kernel, then `_fix_waveform_shape`; D8), `TimeStretch` (caches `phase_advance`
= linspace(0, pi*hop, n_freq)[...,None]; rate selection; raises if no fixed/overriding rate). Tests:
delegation-equivalence to the golden-verified functional ops (exact) + RNG property/reproducibility +
the rate-null raise. `audio_test_transform` 82/82; `ctest` 5/5; `_transform_time.hpp` 100% line +
100% function coverage (1 partial branch; line coverage is the project bar), `_functional.hpp` still
100% lines (G1 helpers covered by the resample test). Decided: `SpeedPerturbation` uses the global RNG
(torchaudio parity), no per-instance seed.

## Objective
Implement the time-domain transforms. The headline work is **kernel caching**: `transform::Resample`
caches the sinc kernel (vs `functional::resample` which rebuilds it each call), which requires
factoring the functional resampler into build/apply halves (G1). `Speed`/`SpeedPerturbation`/
`PitchShift` reuse that machinery.

## Scope
In:
- **G1 functional refactor**: split `functional::resample` into
  `functional::detail::sinc_resample_kernel(orig_freq, new_freq, gcd, opt, dtype, device) -> {kernel,
  width}` + `functional::detail::apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd,
  kernel, width) -> tensor_t`, with `functional::resample` calling both. Public behavior of
  `functional::resample` must stay byte-identical (progress01 golden values must still pass).
- `transform::Resample` — ctor caches `kernel_`/`width_`/`gcd_` (only if `orig!=new`); `operator()`
  short-circuits identity, else applies the cached kernel.
- `transform::Speed` — compose `Resample(source, target)` (source=int(factor*orig), target=orig,
  each /gcd); `operator()(wav[, lengths])` returns `{resampled, out_lengths}` with
  `out_lengths=ceil(lengths*target/source)`.
- `transform::SpeedPerturbation` — holds a vector of `Speed` (one per factor); `operator()` draws a
  factor per call (RNG, optional seed/generator) and dispatches.
- `transform::PitchShift` — eager kernel (D8); `operator()` runs
  stft→phase_vocoder→istft→apply_sinc_resample_kernel→fix-shape via `functional::pitch_shift`'s
  helpers (`_stretch_waveform`, `_fix_waveform_shape` already exist in functional) + the apply
  helper from G1; identity short-circuit when `orig==sample_rate`.
- `transform::TimeStretch` — cache `phase_advance = linspace(0, pi*hop, n_freq)[...,None]`;
  `operator()(complex_spec[, overriding_rate])` → `functional::phase_vocoder`; complex-input check;
  raise if both `fixed_rate` and `overriding_rate` are unset.
Out:
- Masking/Fade (task22). Beamforming (task25).

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress02-...md` — G1 (kernel split), D8 (eager kernel), G5 (RNG).
2. `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — `resample`, `speed`, `pitch_shift`,
   `phase_vocoder`, and the existing private `_stretch_waveform`/`_fix_waveform_shape` helpers; the
   internal resample kernel code (to factor out).
3. torchaudio v2.5.1 `transforms/_transforms.py`: `Resample`, `Speed`, `SpeedPerturbation`,
   `PitchShift`, `TimeStretch`; and `functional/functional.py` `_get_sinc_resample_kernel` /
   `_apply_sinc_resample_kernel` (the shapes/args to mirror in the G1 split).

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — the G1 refactor (keep `resample`
  public signature/output identical).
- NEW `_transform_time.hpp` (or fold into `_transform_augment.hpp` per D6) for the five classes.
- `_transform.hpp`, `unit_test/audio/transform/{main.cpp, gen_golden.py}`.

## Deliverables
- The G1 build/apply helpers in `functional::detail`, with `functional::resample` re-expressed on
  top of them (no behavior change; re-run progress01 resample golden to confirm).
- The five classes in `torchmedia::audio::transform`, each with its own `xxx_option`:
  - `Resample{ kernel_, width_, gcd_, orig_, new_; operator()(wav) }` — identity short-circuit.
  - `Speed{ Resample resampler_; source_, target_; operator()(wav, lengths=nullopt) ->
    std::pair<tensor_t, c10::optional<tensor_t>> }`.
  - `SpeedPerturbation{ std::vector<Speed> speeders_; operator()(wav, lengths=nullopt) }` — RNG
    factor pick; optional `int64_t seed`/generator in the option for deterministic tests.
  - `PitchShift{ kernel_, width_, gcd_, window_, ...; operator()(wav) }`.
  - `TimeStretch{ phase_advance_; fixed_rate_; operator()(complex_spec, overriding_rate=nullopt) }`.
- Golden / property tests; golden block in `gen_golden.py`.

## Steps
1. **G1 refactor first.** Extract the kernel build + apply from `functional::resample` into
   `functional::detail::{sinc_resample_kernel, apply_sinc_resample_kernel}`. Rebuild & re-run the
   existing `audio_test_functional` resample golden — must stay green before touching transforms.
2. **`Resample`** — ctor: if `orig==new`, mark identity; else compute `gcd_` and
   `{kernel_, width_} = detail::sinc_resample_kernel(orig, new, gcd_, opt, dtype, device)`.
   `operator()`: identity returns input; else `detail::apply_sinc_resample_kernel(wav, orig, new,
   gcd_, kernel_, width_)`.
3. **`Speed`** — compute `source_/target_`; build `resampler_ = Resample(source_, target_)`;
   `operator()` returns `{resampler_(wav), out_lengths}` (out_lengths only when lengths given).
4. **`SpeedPerturbation`** — build one `Speed` per factor; `operator()` draws
   `idx = torch::randint(speeders_.size(), {}, gen)` and dispatches. Property test: output shape
   matches one of the candidate speeds; with a fixed seed the pick is reproducible.
5. **`PitchShift`** — eager-build the resample kernel (D8) for `orig_freq=int(sample_rate/rate)`,
   `rate=2^(-n_steps/bins_per_octave)`; `operator()` mirrors `functional::pitch_shift` but uses the
   cached kernel via the apply helper. Simplest correct path: call `functional::pitch_shift` and,
   if the cache makes a measurable difference, switch to the helper — but the deliverable is the
   cached path. Identity short-circuit when `orig==sample_rate`.
6. **`TimeStretch`** — cache `phase_advance_`; `operator()` selects `rate` (overriding vs fixed,
   raise if both null), warns on non-complex input, returns `functional::phase_vocoder(spec, rate,
   phase_advance_)`.
7. **Tests + golden + ctest + coverage** — golden for `Resample`/`PitchShift`/`TimeStretch`;
   property/shape + fixed-seed for `Speed`/`SpeedPerturbation`. Coverage 100% incl. identity
   short-circuits and the rate-null raise.

## Acceptance criteria
- [ ] `functional::resample` refactored onto `detail::sinc_resample_kernel`/`apply_...`; progress01
      resample golden still green (behavior unchanged).
- [ ] `transform::Resample` caches the kernel (construct once, call twice → identical, no rebuild)
      and short-circuits identity.
- [ ] `Speed` returns correct `out_lengths`; `SpeedPerturbation` is reproducible with a fixed seed.
- [ ] `PitchShift`/`TimeStretch` match baked torchaudio 2.5.1 golden values (atol 1e-4).
- [ ] `TimeStretch` raises when neither fixed nor overriding rate is set.
- [ ] `ctest --test-dir build` green; 100% line coverage of the new header + the G1-touched lines.

## Constraints
- Header-only, ATen only. The G1 split must not change `functional::resample`'s observable output.
- Eager kernel (D8) — no LazyModuleMixin analogue.
- Keep `clang-format` clean.

## Notes / Assumptions
- Assumption: `functional::pitch_shift` and its `_stretch_waveform`/`_fix_waveform_shape` helpers
  are reusable as-is; only the resample-apply step swaps to the cached kernel.
- Assumption: float32 working dtype; the cached kernel follows the ctor dtype (document precision
  loss if a higher-precision input is later passed, matching torchaudio's note).
- Dependency: independent of task19/20. The G1 refactor touches functional — coordinate so it lands
  cleanly (it is additive: a private split, public API unchanged).
- Question for Mux: `SpeedPerturbation` seeding policy — accept an optional seed in the option for
  deterministic tests (recommended), or rely on the global torch RNG?
