# Task 10 — Implement SpecAugment masks (mask_along_axis, mask_along_axis_iid)
id: 2026-06-01/task10
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Add torch-native `mask_along_axis` and `mask_along_axis_iid` (plus the shared `_get_mask_param`
helper) to `_audio/_functional.hpp`, mirroring torchaudio v2.5.1's SpecAugment time/frequency masking.

## Scope
In:
- `mask_along_axis(specgram, mask_param, mask_value, axis, p=1.0)` — single mask shared across the
  packed batch.
- `mask_along_axis_iid(specgrams, mask_param, mask_value, axis, p=1.0)` — independent per-example masks.
- `_get_mask_param(mask_param, p, axis_length)` inline helper shared by both.
- Assertion tests (property/shape + fixed-seed interval checks) in `unit_test/audio/functional/main.cpp`.
Out:
- The `torchaudio.transforms.{FrequencyMasking,TimeMasking,SpecAugment}` wrappers (transform layer —
  later progress).
- Autograd/backward (masking is a `masked_fill`; no custom grad needed, but no grad tests required).
- Any non-RNG "iid" optimization or batched-RNG matching with Python torch (exact value-match only
  under matched RNG — see Notes).

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D1 (.venv golden), D5 (RNG ops use
   property/shape tests), D6 (task10 is independent / any order).
2. `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/functional.py` —
   authoritative source for `_get_mask_param`, `mask_along_axis`, `mask_along_axis_iid` (verbatim logic
   reproduced in Steps below).

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — append the three functions (after
  `resample`); reuse `handle_exceptions<torch::Tensor, std::invalid_argument>(...)` and the
  `tensor_t` / `const_tensor_lref_t` aliases already used in this header.
- `unit_test/audio/functional/main.cpp` — add tests + register in `main()`'s call list.
- `unit_test/audio/functional/gen_golden.py` — extend only if a fixed-seed cross-check is added.
- (No new option struct needed — args are plain scalars; do NOT touch
  `_functional_methods_options.hpp`.)

## Deliverables
- In `_functional.hpp`, three inline free functions in `torchmedia::audio::functional`:
  - `inline auto _get_mask_param(int64_t mask_param, double p, int64_t axis_length) -> int64_t;`
  - `inline auto mask_along_axis(const_tensor_lref_t specgram, int64_t mask_param, double mask_value,
    int64_t axis, double p = 1.0) -> tensor_t;`
  - `inline auto mask_along_axis_iid(const_tensor_lref_t specgrams, int64_t mask_param, double mask_value,
    int64_t axis, double p = 1.0) -> tensor_t;`
- Tests `test_mask_along_axis*` in `main.cpp` registered in `main()`, run via the existing
  `audio_test_functional` ctest target; new golden constants (if any) baked into `main.cpp`.

## Steps
1. **`_get_mask_param` helper** — return `mask_param` if `p == 1.0`; else
   `std::min(mask_param, (int64_t)std::floor(axis_length * p))` (Python `int(axis_length * p)` truncates
   toward zero; `axis_length`/`p` are non-negative here, so `floor` matches). Used by both callers
   before any RNG.
2. **`mask_along_axis` (shared single mask)** —
   - `dim = specgram.dim()`. Validate via `handle_exceptions`/`TORCH_CHECK`: `dim >= 2`
     ("Spectrogram must have at least two dimensions ..."); `axis == dim-2 || axis == dim-1`
     ("Only Frequency and Time masking are supported ..."); `0.0 <= p <= 1.0`.
   - `mask_param = _get_mask_param(mask_param, p, specgram.size(axis))`; if `< 1` return `specgram`
     unchanged.
   - Save `shape = specgram.sizes().vec()`; pack: `auto sg = specgram.reshape({-1, shape[-2], shape[-1]})`.
   - `int64_t apack = axis - dim + 3` (axis index after packing to 3-D); `axis_len = sg.size(apack)`.
   - RNG: `auto value = torch::rand({1}) * mask_param;` `auto min_value = torch::rand({1}) *
     (axis_len - value);`.
   - `mask_start = min_value.to(torch::kLong).squeeze();`
     `mask_end = (min_value.to(torch::kLong) + value.to(torch::kLong)).squeeze();` (scalar tensors).
   - `auto mask = torch::arange(0, axis_len, sg.options());`
     `mask = (mask >= mask_start).logical_and(mask < mask_end);` if `axis == dim-2` then
     `mask = mask.unsqueeze(-1)` (frequency mask broadcasts over time).
   - Raise if `(mask_end - mask_start).item<int64_t>() >= mask_param`
     ("Number of columns to be masked should be less than mask_param").
   - `sg = sg.masked_fill(mask, mask_value);` then reshape back: rebuild the output shape as
     `shape[:-2]` ++ `sg.sizes()[-2:]` and `return sg.reshape(out_shape)`.
3. **`mask_along_axis_iid` (per-example masks)** —
   - `dim = specgrams.dim()`. Validate: `dim >= 3`
     ("Spectrogram must have at least three dimensions ..."); same `axis` and `p` checks as above.
   - `mask_param = _get_mask_param(mask_param, p, specgrams.size(axis))`; if `< 1` return unchanged.
   - Batch-dims shape = `specgrams.sizes().slice(0, dim-2)` (everything before the last two dims);
     `axis_len = specgrams.size(axis)`.
   - RNG per example: `auto value = torch::rand(batch_dims, specgrams.options()) * mask_param;`
     `auto min_value = torch::rand(batch_dims, specgrams.options()) * (axis_len - value);`.
   - `auto mask_start = min_value.to(torch::kLong).unsqueeze(-1).unsqueeze(-1);` (shape `(...,1,1)`);
     `auto mask_end = (min_value.to(torch::kLong) + value.to(torch::kLong)).unsqueeze(-1).unsqueeze(-1);`.
   - `auto mask = torch::arange(0, axis_len, specgrams.options());` (`(axis_len,)`).
   - Move the target axis to last so the 1-D `mask` broadcasts:
     `auto out = specgrams.transpose(axis, -1);`
     `out = out.masked_fill((mask >= mask_start).logical_and(mask < mask_end), mask_value);`
     `out = out.transpose(axis, -1);` `return out;`.
4. **Tests, golden, coverage** — add to `main.cpp` (RNG-dependent → property/invariant first, per D5):
   - Validation: feed a 1-D tensor / bad `axis` / `p` outside `[0,1]` and assert the function throws
     `std::invalid_argument` (matches torchaudio's `ValueError` surface).
   - `_get_mask_param`: `p==1.0` returns `mask_param`; `p<1` clamps to `floor(axis_len*p)` (e.g.
     `axis_len=10, p=0.3, mask_param=8 -> 3`).
   - Early return: `mask_param` clamped `< 1` returns the input unchanged (`TM_CHECK_TENSOR_CLOSE`
     equal to input).
   - Fixed-seed interval check: `torch::manual_seed(0)` before the call, then recompute the expected
     `[start,end)` from the same RNG draws and assert exactly those rows/cols equal `mask_value` and the
     rest are untouched. Cover `axis == dim-1` (time, no unsqueeze) and `axis == dim-2`
     (frequency, unsqueeze branch) for `mask_along_axis`.
   - `mask_along_axis_iid`: build a `(B,F,T)` input with `B>=2`; after masking assert the masked
     index sets differ across at least two examples (per-example masks), and that masked positions hold
     `mask_value` while unmasked positions are unchanged.
   - Optional cross-check: only if RNG is matched to Python — torchaudio uses `torch.rand(1)` /
     `torch.rand(batch_dims)`; matching the libtorch global generator to `.venv` is brittle, so prefer
     invariants. If added, extend `gen_golden.py` and bake the constants into `main.cpp`.
   - Register `test_mask_along_axis_*` in `main()`; `ctest --test-dir build` green; confirm 100% line
     coverage of the new lines in `_functional.hpp` (every branch: both `axis` cases, the early-return,
     the `mask_param`-clamp, and the `>= mask_param` raise).

## Acceptance criteria
- [ ] `mask_along_axis` / `mask_along_axis_iid` / `_get_mask_param` compile inline in `_functional.hpp`
      with the signatures above and no new runtime deps.
- [ ] Validation raises match torchaudio (dim, axis, p) and surface as `std::invalid_argument`.
- [ ] Under a fixed `torch::manual_seed`, the masked interval equals the formula-derived `[start,end)`;
      `axis=dim-1` vs `axis=dim-2` and the `p<1` clamp behave correctly; the iid variant produces
      per-example-distinct masks.
- [ ] Early return (`mask_param < 1` after clamp) returns the input unchanged.
- [ ] `ctest --test-dir build` green; 100% line coverage of the newly added `_functional.hpp` lines
      (vendored `_vendor/` excluded).

## Constraints
- Header-only; inline free functions in `torchmedia::audio::functional`; torch-native ATen ops only
  (`torch::rand`, `.to(torch::kLong)`, `torch::arange`, `.reshape`, `.transpose`, `.masked_fill`,
  `.logical_and`, `.unsqueeze`).
- Match torchaudio's validation order and messages; replicate the exact packed-axis index
  `axis - dim + 3` and the `axis == dim-2` unsqueeze (frequency) vs no-unsqueeze (time) branch.
- RNG-dependent: do NOT assert exact value-equality with the `.venv` unless the global generator is
  explicitly matched; rely on invariants + fixed-seed self-derived expectations (D5).
- No option struct (plain scalar args); do not modify `_functional_methods_options.hpp`.

## Notes / Assumptions
- Assumption: independent of task01 (`lfilter`) and all other tasks — needs only the existing
  `_functional.hpp` scaffolding (`handle_exceptions`, type aliases) already in the repo.
- Assumption: torchaudio's `mask_along_axis` raise `mask_end - mask_start >= mask_param` is effectively
  unreachable under normal RNG (value < mask_param), but reproduce it for parity; cover it in a test by
  constructing the condition directly if needed (otherwise note it as a defensive branch and exclude via
  a targeted test that forces a small `mask_param`).
- Assumption: `mask_value` is `double` (torchaudio passes a Python float); cast at `masked_fill`.
- Question: confirm with Mux whether a matched-RNG `.venv` golden is wanted for these ops, or whether
  property/invariant + fixed-seed self-reference tests (the D5 default) are sufficient — recommend the
  latter to keep the test portable and non-flaky.
