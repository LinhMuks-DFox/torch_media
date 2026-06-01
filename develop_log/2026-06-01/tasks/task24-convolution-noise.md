# Task 24 — Convolution & noise (Convolve, FFTConvolve, AddNoise)
id: 2026-06-01/task24
parent: 2026-06-01/progress02
status: done              # active | blocked | done
owner: code_agent

## Result
Done 2026-06-01. Added `Convolve`, `FFTConvolve` (both carry a `convolve_option` reusing the existing
`functional::convolve_mode` enum {full,valid,same}) and `AddNoise` (no stored config) to
`_transform_effects.hpp`. Each forwards to `functional::convolve` / `fftconvolve` / `add_noise`.
Tests: delegation-equivalence for all three convolve modes + AddNoise with and without the optional
`lengths`. `audio_test_transform` 133/133; `ctest` 5/5; `_transform_effects.hpp` 100% line/function
coverage. clang-format clean.

## Objective
Implement the convolution/noise transforms: `Convolve`, `FFTConvolve` (both store a `mode` enum and
forward to the matching functional op), and `AddNoise` (no config — a pure forward over
`functional::add_noise`).

## Scope
In:
- `Convolve(mode=full)` → `functional::convolve(x, y, mode)`; `mode` is the existing
  `functional::convolve_mode` enum {full, valid, same} (reuse it — D5 says enums, and this one
  already exists).
- `FFTConvolve(mode=full)` → `functional::fftconvolve(x, y, mode)`.
- `AddNoise()` — no stored config; `operator()(waveform, noise, snr, lengths=nullopt)` →
  `functional::add_noise(waveform, noise, snr, lengths)`.
Out:
- Everything else.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress02-...md` — D5 (enums), D1 (class idiom).
2. `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — `convolve`, `fftconvolve`,
   `add_noise` signatures and the `convolve_mode` enum.
3. torchaudio v2.5.1 `transforms/_transforms.py`: `Convolve`, `FFTConvolve`, `AddNoise` (note the
   `_check_convolve_mode` validation and that these are binary ops taking two signals).

Code to inspect/change:
- `_transform_effects.hpp` (shared with task23; add these three).
- `_transform.hpp`, `unit_test/audio/transform/{main.cpp, gen_golden.py}`.

## Deliverables
- The three classes in `torchmedia::audio::transform`:
  - `Convolve{ convolve_mode mode_; tensor_t operator()(const_tensor_lref_t x, const_tensor_lref_t
    y) const; }`.
  - `FFTConvolve` — same shape.
  - `AddNoise{ tensor_t operator()(const_tensor_lref_t waveform, const_tensor_lref_t noise,
    const_tensor_lref_t snr, const c10::optional<tensor_t>& lengths = c10::nullopt) const; }`.
- Golden cross-checks for all three; golden block in `gen_golden.py`.

## Steps
1. **`Convolve`/`FFTConvolve`** — ctor stores the `convolve_mode`; `operator()(x, y)` forwards.
   (Mode is already an enum in functional, so no string validation needed — invalid states are
   unrepresentable.)
2. **`AddNoise`** — no ctor state; `operator()` forwards the four args.
3. **Tests + golden + ctest + coverage** — golden vs `T.Convolve/FFTConvolve/AddNoise` for each
   `mode`; for `AddNoise`, a fixed waveform/noise/snr and the optional-`lengths` path. Coverage 100%
   incl. each `mode` and the lengths-present/absent branch.

## Acceptance criteria
- [ ] Three classes exist, `torchmedia::audio::transform`, header-only, delegating to functional.
- [ ] `Convolve`/`FFTConvolve` carry the `convolve_mode` enum (no runtime string).
- [ ] All three match baked torchaudio 2.5.1 golden values (atol 1e-5).
- [ ] `AddNoise` exercises both the lengths-given and lengths-absent paths.
- [ ] `ctest --test-dir build` green; 100% line coverage of the new lines.

## Constraints
- Header-only, ATen only. No new .cpp. Delegate to functional; do not reimplement.
- Reuse the existing `functional::convolve_mode` enum.
- Keep `clang-format` clean.

## Notes / Assumptions
- Assumption: `functional::add_noise` already validates shapes/SNR; the transform adds nothing.
- Dependency: independent of all other tasks.
- Question for Mux: none — these are firmly in scope and unambiguous.
