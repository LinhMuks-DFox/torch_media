# Task 06 — Companding & emphasis: mu_law ×2, preemphasis, deemphasis
id: 2026-06-01/task06
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Add torch-native `mu_law_encoding`, `mu_law_decoding`, `preemphasis`, and `deemphasis` to
`_audio/_functional.hpp` (`deemphasis` calling task01's `lfilter`), mirroring torchaudio v2.5.1, each
with assertion + golden tests and 100% line coverage.

## Scope
In:
- `mu_law_encoding(waveform, quantization_channels=256)` — float/int → integer mu-law codes `0..mu`.
- `mu_law_decoding(x_mu, quantization_channels=256)` — integer codes → float waveform in `[-1, 1]`.
- `preemphasis(waveform, coeff=0.97)` — single-tap FIR pre-emphasis.
- `deemphasis(waveform, coeff=0.97)` — IIR inverse of `preemphasis`, implemented via `lfilter`.
Out:
- Autograd/backward (forward only; libtorch tracks grad automatically when inputs require it).
- Any option struct — these four ops take plain scalar args only (no `xxx_option`).
- `lfilter` itself (owned by task01); this task only *consumes* it for `deemphasis`.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — parent progress; D3 (lfilter
   keystone, coeff ordering `[a0,a1,...]`), Tier 1 plan (task06 depends on task01).
2. `develop_log/2026-06-01/tasks/task01-lfilter-biquad-filtfilt.md` — the `lfilter` C++ signature
   and coefficient convention `deemphasis` must call.
3. torchaudio v2.5.1 source (authoritative algorithm):
   `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/functional.py`
   — read `mu_law_encoding`, `mu_law_decoding`, `preemphasis`, `deemphasis`.

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — append the four inline functions
  (follow `db_to_amplitude`/`convolve` style already there).
- `libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp` — for the `lfilter`
  declaration `deemphasis` calls (created by task01; this header is aggregated by `torchmedia.hpp`).
- `unit_test/audio/functional/main.cpp` — add tests + register in `main()`'s call list.
- `unit_test/audio/functional/gen_golden.py` — add the torchaudio golden constants.

## Deliverables
- In `_functional.hpp`, namespace `torchmedia::audio::functional`, inline free functions:
  - `inline auto mu_law_encoding(tensor_t x, int64_t quantization_channels = 256) -> tensor_t;`
  - `inline auto mu_law_decoding(tensor_t x_mu, int64_t quantization_channels = 256) -> tensor_t;`
  - `inline auto preemphasis(tensor_t waveform, double coeff = 0.97) -> tensor_t;`
  - `inline auto deemphasis(tensor_t waveform, double coeff = 0.97) -> tensor_t;`
- Tests in `main.cpp` (`test_mu_law_roundtrip`, `test_mu_law_golden`, `test_preemphasis`,
  `test_deemphasis_roundtrip`) registered in `main()`.
- New golden constants appended to `gen_golden.py` (mu-law codes on a small ramp, preemphasis output,
  deemphasis round-trip) and baked into `main.cpp`.

## Steps
1. **mu_law_encoding** — `mu = quantization_channels - 1`. If `x.is_floating_point()` is false,
   `x = x.to(torch::kFloat)`. Compute `x_mu = torch::sign(x) * torch::log1p(mu * torch::abs(x)) /
   std::log1p(double(mu))` (numerator uses `torch::log1p`; the denominator is a host scalar via
   `std::log1p`). Return `((x_mu + 1) / 2 * mu + 0.5).to(torch::kInt64)` (integer codes in `0..mu`).
2. **mu_law_decoding** — `mu = quantization_channels - 1`. Cast `x_mu` to float
   (`x_mu.to(torch::kFloat)`). `x = (x_mu / mu) * 2 - 1`. Return
   `torch::sign(x) * (torch::exp(torch::abs(x) * std::log1p(double(mu))) - 1) / mu` (float in
   `[-1, 1]`). Use `std::log1p(mu)` for the host scalar.
3. **preemphasis (true FIR, not recursive)** — clone the input: `auto w = waveform.clone();`. Build
   the shifted term as a *materialized* tensor first so the subtract reads the original samples, not
   in-place-updated ones: `auto shifted = coeff * w.index({..., Slice(None, -1)});` then
   `w.index({..., Slice(1, None)}).sub_(shifted);` (use `torch::indexing::{Slice, None}` and
   `Ellipsis`). Element 0 along the last axis is left unchanged. Return `w`. Note: the multiply
   creating `shifted` is what makes this a genuine FIR `y[i] = x[i] - coeff*x[i-1]` rather than the
   recursive IIR form.
4. **deemphasis (IIR inverse via lfilter)** — construct
   `b_coeffs = torch::tensor({1.0, 0.0})` and `a_coeffs = torch::tensor({1.0, -coeff})` (dtype/device
   matching `waveform`), then return `lfilter(waveform, a_coeffs, b_coeffs, /*clamp=*/false)` using
   task01's signature. This exactly inverts `preemphasis` (the FIR `[1, -coeff]` numerator becomes the
   IIR denominator). Add `#include "_functional_filtering.hpp"` to `_functional.hpp` only if task01's
   `lfilter` lives there and is not already visible.
5. **Validation parity** — match torchaudio: no extra raises for these ops beyond dtype handling
   (mu-law accepts non-float via cast). Use `TORCH_CHECK`/`handle_exceptions` only if torchaudio
   raises; otherwise none.
6. **Tests + golden + green** — extend `gen_golden.py`:
   `F.mu_law_encoding(torch.linspace(-1,1,9), 256)`, `F.mu_law_decoding(<those codes>, 256)`,
   `F.preemphasis(torch.tensor([1.,2.,3.,4.]), 0.97)`,
   `F.deemphasis(F.preemphasis(x,0.97),0.97)` (should ≈ x). Run
   `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`,
   bake printed constants into `main.cpp`. Add: (a) mu-law golden codes match; (b)
   `mu_law_decoding(mu_law_encoding(x)) ≈ x` round-trip within companding tolerance; (c) preemphasis
   closed-form on `[1,2,3,4]` → `[1, 2-0.97*1, 3-0.97*2, 4-0.97*3]`; (d)
   `deemphasis(preemphasis(x)) ≈ x` round-trip. Build & run:
   `cmake --build build --target audio_test_functional &&
   ./build/unit_test/audio/functional/audio_test_functional`; then `ctest --test-dir build` green;
   confirm 100% line coverage of the four new functions.

## Acceptance criteria
- [ ] `mu_law_encoding` output equals torchaudio golden integer codes on the 9-point ramp (exact).
- [ ] `mu_law_decoding(mu_law_encoding(x))` ≈ `x` within companding round-trip tolerance (atol ≈ 5e-3).
- [ ] `preemphasis([1,2,3,4], 0.97)` matches the closed-form `[1, 1.03, 1.06, 1.09]` (atol 1e-6).
- [ ] `deemphasis(preemphasis(x))` ≈ `x` (atol 1e-5) and matches torchaudio `deemphasis` golden.
- [ ] `ctest --test-dir build` green; 100% line coverage of the four new functions
      (`--ignore-filename-regex='_vendor/.*'`).

## Constraints
- Header-only: inline free functions in `_functional.hpp`, namespace
  `torchmedia::audio::functional`; no `.cpp`.
- Torch-native ATen only: `torch::sign`, `torch::log1p`, `torch::abs`, `torch::exp`, `.to(...)`,
  `torch::tensor`, indexing `Slice`/`None`/`Ellipsis`, `sub_`; host scalars via `std::log1p`.
- Match torchaudio v2.5.1 numerics and (absence of) raises exactly.
- `deemphasis` is the only op with a cross-task dependency — it must call task01's `lfilter`
  (sequential IIR recurrence) and must not reimplement the recurrence locally.

## Notes / Assumptions
- Assumption: `tensor_t` is the project alias for `torch::Tensor` (as used by `convolve`/
  `db_to_amplitude` in `_functional.hpp`); reuse it.
- Assumption: task01 lands `lfilter` with signature roughly
  `lfilter(waveform, a_coeffs, b_coeffs, bool clamp=true, bool batching=true)` and coeff order
  `[a0,a1,...]`; if the real signature differs, adapt the `deemphasis` call site to it.
- Gotcha (preemphasis): materialize `coeff * w[..., :-1]` into a temporary *before* the in-place
  `sub_` on `w[..., 1:]` — doing the multiply lazily/in-place would turn the FIR into a recursive
  filter and break the `deemphasis` round-trip.
- Gotcha (mu_law_encoding): the `+ 0.5` then `.to(kInt64)` is truncation-based rounding to nearest;
  keep the order exactly as torchaudio to match codes at the bin edges.
- Dependency: this task is **blocked on task01** for `deemphasis` only. The three torch-native ops
  (`mu_law_encoding`, `mu_law_decoding`, `preemphasis`) can be implemented and merged independently;
  if task01 is not yet done, land those three and mark the `deemphasis` test as the remaining item.
- Question for Mux: none — scope is fully within the agreed functional surface (D1/D3).
