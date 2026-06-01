# Task 03 — Implement simple SoX effects: contrast, dcshift, gain
id: 2026-06-01/task03
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Add the three stateless, IIR-free SoX effects `contrast`, `dcshift`, and `gain` to the new
`_functional_filtering.hpp` header (torch-native, header-only), each with closed-form assertion tests.

## Scope
In:
- `contrast(waveform, enhancement_amount = 75.0)` — sinusoidal contrast enhancement.
- `dcshift(waveform, shift, limiter_gain = std::nullopt)` — DC offset shift, with optional soft limiter.
- `gain(waveform, gain_db = 1.0)` — apply a dB gain.
Out:
- Any IIR / `lfilter`-based effect (overdrive, phaser, flanger → task04; biquads → task01/task02).
- Autograd / custom backward (these are plain differentiable elementwise ops; nothing extra needed).
- Compressed-codec or sox round-trip behavior; no batching changes beyond what torchaudio does
  (the ops already broadcast naturally over leading dims).
- `enhancement_amount`/`shift`/`gain_db` as tensors — match torchaudio: plain scalars.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D1 (.venv golden source),
   D2 (file split: these live in `_functional_filtering.hpp`), D5 (testing/coverage rule), D6 (task03
   is independent / any order — no dependency on task01).
2. torchaudio v2.5.1 reference (authoritative algorithm):
   `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/filtering.py`
   — see `contrast`, `dcshift`, `gain` (load WebFetch via ToolSearch `select:WebFetch` if you need the
   exact source; the algorithm is also restated in Steps below).

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp` — NEW header (this task creates
  it if task01 has not yet; otherwise append). Namespace `torchmedia::audio::functional`.
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — pattern reference for `inline auto`
  free-function style, `handle_exceptions<torch::Tensor, std::invalid_argument>(...)` usage,
  `using namespace torch::indexing;`.
- `libtorchmedia/include/torchmedia/globel_include.hpp` — `tensor_t = torch::Tensor`,
  `handle_exceptions<T, ExceptionT>(default_ret, msg)` signature.
- `libtorchmedia/include/torchmedia.hpp` (and `torchmedia/audio.hpp`) — aggregate the new header so it
  is reachable from the public include.
- `unit_test/audio/functional/main.cpp` — add tests + register in `main()`.
- `unit_test/audio/functional/gen_golden.py` — only if you choose the optional torchaudio cross-check.

## Deliverables
- In `_functional_filtering.hpp`, three `inline auto` free functions in
  `torchmedia::audio::functional` (header-only; no `.cpp`):
  - `inline auto contrast(tensor_t waveform, double enhancement_amount = 75.0) -> tensor_t;`
  - `inline auto dcshift(tensor_t waveform, double shift, std::optional<double> limiter_gain = std::nullopt) -> tensor_t;`
  - `inline auto gain(tensor_t waveform, double gain_db = 1.0) -> tensor_t;`
  No new `xxx_option` struct is required (scalar args are few); do NOT add option structs unless you
  find a real need — keep parity with torchaudio's flat signatures.
- The new header included from `torchmedia/audio.hpp` (or wherever `_functional.hpp` is aggregated).
- Assertion tests in `unit_test/audio/functional/main.cpp` (`test_contrast`, `test_dcshift`,
  `test_gain`) registered in `main()`'s call list; covers every branch (see Steps 5).
- (Optional) golden constants in `gen_golden.py` if you add a torchaudio cross-check; bake printed
  values into `main.cpp` (no runtime .venv dependency).

## Steps
1. **Create/locate the header** — if `_functional_filtering.hpp` does not exist, create it with the
   `#pragma once` + include-guard style of `_functional_methods_options.hpp`, include
   `"../globel_include.hpp"`, open `namespace torchmedia::audio::functional`. If task01 already created
   it, append after the existing content.
2. **contrast** — validate `enhancement_amount` in `[0.0, 100.0]`; on violation call
   `handle_exceptions<torch::Tensor, std::invalid_argument>(torch::empty({1}), "...")`. Then:
   `contrast = enhancement_amount / 750.0;`
   `temp1 = waveform * (M_PI / 2.0);`  (use `M_PI`)
   `temp2 = contrast * torch::sin(temp1 * 4.0);`
   `return torch::sin(temp1 + temp2);`  — all `torch::sin` elementwise.
3. **dcshift** — first `auto out = waveform.clone();` to avoid aliasing the input.
   - No-limiter path (`!limiter_gain.has_value()`): `return (out + shift).clamp(-1.0, 1.0);`
   - Limiter path: `double limiter_threshold = 1.0 - (std::abs(shift) - *limiter_gain);`
     - `shift > 0`: for samples where `out > limiter_threshold`, soft-compress; torchaudio:
       `out[mask] = (out[mask] - limiter_threshold) * limiter_gain / (1 - limiter_threshold) + limiter_threshold + shift`
       and elsewhere `out = out + shift`. Implement with `torch::where(mask, compressed, out + shift)`
       (mask = `out > limiter_threshold`), then clamp is NOT applied in the limiter branch (match
       torchaudio — verify against the source; replicate exactly whatever the source does, including any
       final clamp/no-clamp).
     - `shift < 0`: symmetric — mask `out < -limiter_threshold`,
       `out[mask] = (out[mask] + limiter_threshold) * limiter_gain / (1 - limiter_threshold) - limiter_threshold + shift`,
       else `out + shift`.
     - `shift == 0`: torchaudio still returns `out + shift` (== `out`); keep parity.
     Prefer `torch::where` over in-place masked index assignment (cleaner, no aliasing). Use
     `using namespace torch::indexing;` if you do go the masked-assign route.
4. **gain** — `if (gain_db == 0.0) return waveform;` else
   `double ratio = std::pow(10.0, gain_db / 20.0); return waveform * ratio;`
5. **Aggregate + tests** — add `#include "_audio/_functional_filtering.hpp"` to the audio aggregator so
   `#include <torchmedia.hpp>` reaches it. Then add to `main.cpp`:
   - `test_contrast`: closed-form. e.g. for `waveform = [0.0, 0.5, 1.0, -0.5]` and the default
     `enhancement_amount = 75.0` (→ contrast = 0.1), compute expected with the exact formula in C++ (or
     bake a Python golden) and `TM_CHECK_TENSOR_CLOSE`. Also assert the out-of-range raise path
     (`enhancement_amount = 150.0` / `-1.0`) is exercised (call it; the helper returns a sentinel — at
     minimum cover the validation line).
   - `test_dcshift`: (a) no-limiter clamps: `dcshift([0.5, -0.9, 0.8], 0.3)` →
     `clamp([0.8, -0.6, 1.1], -1, 1) = [0.8, -0.6, 1.0]`; (b) limiter `shift > 0`: a sample above
     threshold + a sample below → check both the compressed and the `+shift` element; (c) limiter
     `shift < 0`: symmetric. Use closed-form expected values computed by hand or via gen_golden.py.
   - `test_gain`: `gain_db = 0` returns input unchanged (`TM_CHECK_TENSOR_CLOSE(out, in, 0, 0)`);
     `gain_db = 6.0206` → ratio ≈ 2.0 (`TM_CHECK_TENSOR_CLOSE(gain(x,6.0206), x*2, 1e-4, 1e-5)`);
     a negative dB case (e.g. `-6.0206` → ×0.5).
   - Register `test_contrast(); test_dcshift(); test_gain();` in `main()` before
     `return tm_test::summary(...)`.
   - (Optional cross-check) extend `gen_golden.py` to print
     `F.contrast/F.dcshift/F.gain` outputs for the chosen inputs; run
     `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`
     and bake the constants.
6. **Build, run, coverage** — `cmake --build build --target audio_test_functional &&
   ./build/unit_test/audio/functional/audio_test_functional`; `ctest --test-dir build` green; confirm
   100% line coverage of the three new functions (every branch: contrast valid/invalid; dcshift
   no-limiter / limiter shift>0 / shift<0 / shift==0; gain ==0 / !=0) per the coverage workflow in
   CLAUDE.md (`--ignore-filename-regex='_vendor/.*'`). Run clang-format on touched files.

## Acceptance criteria
- [ ] `contrast`, `dcshift`, `gain` exist as `inline auto` free functions in
      `_functional_filtering.hpp` under `torchmedia::audio::functional` with the signatures above.
- [ ] The new header is reachable via `#include <torchmedia.hpp>` (aggregated).
- [ ] `contrast` output matches the closed-form formula (and torchaudio if cross-checked) within
      `atol=1e-5`; out-of-range `enhancement_amount` triggers the validation path.
- [ ] `dcshift` matches torchaudio/closed-form on: no-limiter clamp, limiter `shift>0`, limiter
      `shift<0` (and `shift==0` no-op) within `atol=1e-5`.
- [ ] `gain` returns input unchanged for `gain_db==0`, and `waveform * 10^(gain_db/20)` otherwise
      (verified at +6.02 dB ≈ ×2 and a negative dB case).
- [ ] `ctest --test-dir build` is green; 100% line coverage of the three new functions (all branches).
- [ ] Touched files clang-formatted.

## Constraints
- Header-only, `inline` free functions; torch-native ATen ops only (`torch::sin`, `torch::clamp`,
  `torch::where`, `tensor.clone()`); no system libs, no `.cpp`.
- Match torchaudio v2.5.1 validation/raises exactly: `contrast` raises on
  `enhancement_amount ∉ [0,100]` (use `handle_exceptions<torch::Tensor, std::invalid_argument>`).
  Replicate the limiter math and any final clamp/no-clamp **exactly as the source does** — verify
  against `filtering.py`, do not infer.
- `dcshift` MUST `clone()` the input before any masked/elementwise mutation to avoid aliasing the
  caller's tensor.
- No `.venv` dependency at build/CI time — golden values, if any, are baked into `main.cpp`.

## Notes / Assumptions
- Assumption: no dependency on task01 (`lfilter`) — these three are pure elementwise/torch ops and may
  be implemented first or in parallel (per progress01 D6: task03 is independent).
- Assumption: if task01 has already created `_functional_filtering.hpp`, append to it and reuse its
  include guard / namespace block rather than re-opening a second namespace.
- Gotcha: `contrast` uses `enhancement_amount / 750` (not `/75`); the inner sine is `sin(temp1 * 4)`.
  `temp1 = waveform * pi/2`. Double-check both constants against the source.
- Gotcha: `dcshift` limiter uses `limiter_threshold = 1 - (|shift| - limiter_gain)` and divides by
  `(1 - limiter_threshold)`; guard nothing extra unless the source does, but verify the divisor is
  non-zero for the test inputs you pick.
- Gotcha: `gain` early-returns the *same* tensor object on `gain_db == 0` (no clone) — torchaudio does
  not copy here; the test for the zero case should compare value-equality, not identity.
- Question for Mux: none — task03 is in-scope and self-contained; flag only if the `dcshift` limiter
  source diverges from the restated math above (then follow the source and note the deviation in the
  Agent log).
