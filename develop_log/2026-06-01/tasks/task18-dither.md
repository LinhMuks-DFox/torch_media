# Task 18 — Implement dither (TPDF/RPDF/GPDF + optional noise shaping)
id: 2026-06-01/task18
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Add torch-native `dither` (probability-distribution dithering with optional error-feedback noise
shaping) to `_functional_filtering.hpp`, mirroring `torchaudio.functional.dither`.

## Scope
In:
- `dither(waveform, density_function="TPDF", noise_shaping=false)`.
- The two private helpers as inline `_`-prefixed functions: `_apply_probability_distribution`
  (TPDF/RPDF/GPDF) and `_add_noise_shaping`.
Out:
- No new distributions beyond TPDF/RPDF/GPDF (match torchaudio exactly).
- Not a quantization/bit-depth API — dither only adds shaped noise + rounds at the 16-bit scale.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D1 (golden via .venv), D2 (file org).
2. `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/filtering.py`
   — verbatim `dither`, `_apply_probability_distribution`, `_add_noise_shaping` source.

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp` (append `dither` + helpers).
- `unit_test/audio/functional/{main.cpp, gen_golden.py}` (tests + any golden constants).

## Deliverables
- `inline auto dither(const_tensor_lref_t waveform, const std::string &density_function = "TPDF", bool noise_shaping = false) -> tensor_t;`
  in `_functional_filtering.hpp`, namespace `torchmedia::audio::functional`, plus the two inline
  helpers. (A `dither_option` struct is optional — only two flags, so positional args are fine.)
- Assertion tests in `main.cpp` registered in `main()`, ctest target `audio_test_functional` green.

## Steps
1. **_apply_probability_distribution** — scale `signal_scaled = waveform * (2^15 - 2)`. Branch on
   `density_function`:
   - `"TPDF"`: build triangular-distribution noise from `torch::bartlett_window` (the torchaudio
     approach — a Bartlett/triangular window convolved/sampled across time); add to `signal_scaled`.
   - `"RPDF"`: add uniform noise `(waveform[rand_channel][rand_time] - 0.5)` (rectangular).
   - `"GPDF"`: sum 6 random samples for an approximate Gaussian, add to `signal_scaled`.
   Then quantize `torch::round(...)` and rescale `quantised / 2^15`. Raise on an unknown
   `density_function` string (match torchaudio's behavior).
2. **_add_noise_shaping** — error-feedback: `error = dithered_waveform - waveform`; shift the error by
   one sample along time (`torch::cat` a leading zero, drop the last) and add back to `dithered`.
   This is a single vectorized shift-and-add per channel (NOT a per-sample recursion).
3. **dither** — call `_apply_probability_distribution(waveform, density_function)`; if `noise_shaping`,
   pass the result + original through `_add_noise_shaping`; return. Preserve input shape (pack/unpack
   batch with `reshape`/`view` as torchaudio does).
4. **Tests + golden** — RNG-dependent (random noise), so prefer property/shape tests:
   output shape == input shape; dtype preserved; for a constant/zero input the deterministic parts
   (scale, round, noise-shaping shift) are checkable; verify the unknown-`density_function` raise.
   For a point-wise `.venv` cross-check, extend `gen_golden.py` and match `torch::manual_seed` to the
   torchaudio call's RNG; otherwise assert invariants. ctest green; 100% line coverage of the new
   lines (exercise all three distributions + `noise_shaping` true/false + the error branch).

## Acceptance criteria
- [ ] `dither` output shape/dtype match the input across TPDF/RPDF/GPDF and `noise_shaping` on/off.
- [ ] Deterministic sub-behaviors (16-bit scale, `round` quantization, one-sample error shift) match a
      hand-computed C++ self-reference; unknown `density_function` raises.
- [ ] (If RNG matched) point-wise close to `torchaudio.functional.dither` in `.venv`.
- [ ] ctest green; 100% line coverage of the added `dither`/helper lines.

## Constraints
- Header-only, inline, torch-native ATen ops only: `torch::bartlett_window`, `torch::randint`,
  `torch::round`, `torch::zeros`, `torch::cat`, `reshape`/`view`.
- Match torchaudio's distribution math exactly (the `2^15 - 2` scale and `2^15` rescale are precise).
- Independent of the IIR core — no dependency on task01 `lfilter`.

## Notes / Assumptions
- Assumption: `dither` lives in `_functional_filtering.hpp` (it is in torchaudio's `filtering.py`),
  even though it is not an IIR filter.
- Assumption: RNG-dependent output means exact value tests require a matched torch generator seed;
  default to invariant/property tests, mirroring how `gen_golden.py` already handles randomized inputs.
- This task was added after the initial 17 — `dither` was the one `__all__` entry missed by the first
  grouping; it completes coverage of the 53 unported entries.
