# Task 01 — IIR core: lfilter, biquad, filtfilt
id: 2026-06-01/task01
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Result
Done 2026-06-01. `lfilter`/`biquad`/`filtfilt` implemented in NEW `_functional_filtering.hpp`
(included via `audio.hpp`), reusing torchaudio's pure-torch `_lfilter_core` recurrence (depthwise
conv1d FIR + sequential C++ time loop, `/a0` normalisation, clamp, 1D/2D-coeff batching). 7 test
functions added to `unit_test/audio/functional/main.cpp` (identity FIR, one-pole `0.5^n` closed-form,
golden, clamp on/off, 2D batching + `batching=false` stack + raise paths, biquad golden, filtfilt
golden); golden block appended to `gen_golden.py`. `audio_test_functional` 70/70, `ctest` 4/4 green.
Matches torchaudio 2.5.1 point-wise (atol 1e-5). Validation guards use `handle_exceptions` as a bare
statement (throws), matching the existing `convolve` style.

## Objective
Port the keystone IIR primitive `lfilter` plus its thin wrappers `biquad` and `filtfilt` into a NEW
header `libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp`, as torch-native
header-only `inline` free functions in `torchmedia::audio::functional`, with golden-checked tests.

## Scope
In:
- `lfilter(waveform, a_coeffs, b_coeffs, clamp=true, batching=true)` — full reimplementation of
  torchaudio's pure-torch `_lfilter` + `_lfilter_core` + `_lfilter_core_generic_loop` (the sequential
  recurrence). This is the only non-vectorizable piece; everything else is ATen.
- `biquad(waveform, b0, b1, b2, a0, a1, a2)` — UNNORMALIZED-coeff wrapper that builds
  `a_coeffs=[a0,a1,a2]`, `b_coeffs=[b0,b1,b2]` and calls `lfilter` (clamp defaults true).
- `filtfilt(waveform, a_coeffs, b_coeffs, clamp=true)` — forward `lfilter(clamp=false)`, `flip(-1)`,
  `lfilter(clamp=clamp)`, `flip(-1)` back.
- New header `_functional_filtering.hpp`; aggregate it into `audio.hpp`.
Out:
- All 11 biquad **designers** (`lowpass_biquad`, `bass_biquad`, …) — task02; they only need `biquad`.
- `deemphasis`, `loudness` — separate tasks (task06, task12) that depend on this `lfilter`.
- Autograd / custom backward `Function` (torchaudio's `DifferentiableIIR`/`DifferentiableFIR`): out —
  ship a plain forward op (no grad through the recurrence is required by this project).
- The compiled `torch.ops.torchaudio._lfilter` fast path: out — port the pure-Python generic loop only.
- Any scipy-style edge padding in `filtfilt`: out — torchaudio's `filtfilt` does NOT pad; do not add it.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D2 (file split), D3 (lfilter is the
   keystone), D5 (testing), and the `lfilter` gotchas note (coeffs `[a0,a1,…]`, flipped + `/a0`).
2. torchaudio v2.5.1 source (authoritative algorithm):
   `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/filtering.py`
   — functions `_lfilter_core_generic_loop`, `_lfilter_core`, `_lfilter`, `lfilter`, `biquad`,
   `filtfilt`.

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp` — NEW (create it).
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — read for conventions:
  `handle_exceptions<T, ExceptionT>(...)` usage, `convolve_mode`/conv1d usage
  (`torch::nn::functional::conv1d`), `resample` option-struct style.
- `libtorchmedia/include/torchmedia/globel_include.hpp` — `tensor_t`, `const_tensor_lref_t`,
  `handle_exceptions` (throws `ExceptionType` unless `LIB_TORCH_MEDIA_NO_EXCEPTIONS`).
- `libtorchmedia/include/torchmedia/audio.hpp` — add `#include "_audio/_functional_filtering.hpp"`.
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` — only if you choose to
  add a `lfilter_option` (see Notes; default is plain bool args, NO option struct needed here).
- `unit_test/audio/functional/main.cpp` — add tests + register in `main()`'s call list.
- `unit_test/audio/functional/gen_golden.py` — append the golden generator block.

## Deliverables
- In `_functional_filtering.hpp` (header guard `LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_FILTERING_HPP`,
  `#include "../globel_include.hpp"`, namespace `torchmedia::audio::functional`):
  - `inline auto lfilter(const_tensor_lref_t waveform, const_tensor_lref_t a_coeffs,
    const_tensor_lref_t b_coeffs, bool clamp = true, bool batching = true) -> tensor_t;`
  - `inline auto biquad(const_tensor_lref_t waveform, double b0, double b1, double b2, double a0,
    double a1, double a2) -> tensor_t;`
  - `inline auto filtfilt(const_tensor_lref_t waveform, const_tensor_lref_t a_coeffs,
    const_tensor_lref_t b_coeffs, bool clamp = true) -> tensor_t;`
  - (optional, kept private/`static inline`) a `_lfilter_core` helper taking a 3D `(B, C, T)` tensor.
- `audio.hpp` includes the new header (so `torchmedia.hpp` re-exports it transitively).
- Tests in `unit_test/audio/functional/main.cpp` (each registered in `main()`): identity FIR, one-pole
  closed-form, biquad analytic, and torchaudio golden cross-checks for all three functions.
- New golden constants in `gen_golden.py`, baked into `main.cpp` as literals (build must NOT depend on
  `.venv` at runtime).

## Steps
1. **Create the header skeleton** — `_functional_filtering.hpp` with the guard, includes
   (`../globel_include.hpp`, `<torch/nn/functional/conv.h>`, `<torch/nn/options/conv.h>`), and namespace.
   Add `#include "_audio/_functional_filtering.hpp"` to `audio.hpp`.
2. **`_lfilter_core(waveform_3d, a_coeffs_2d, b_coeffs_2d)`** — assume `waveform` is `(n_batch,
   n_channel, n_sample)` and coeffs are `(n_channel, n_order)`. Validate `a_coeffs.sizes() ==
   b_coeffs.sizes()`, `waveform.dim() == 3`, `n_order = a_coeffs.size(1) > 0` (else `handle_exceptions
   <tensor_t, std::invalid_argument>`). Then:
   - `padded_waveform = torch::nn::functional::pad(waveform, F::PadFuncOptions({n_order - 1, 0}))`
     (left-pad `n_order-1` zeros on the time axis); `padded_output = torch::zeros_like(padded_waveform)`.
   - `a_flipped = a_coeffs.flip(1)`, `b_flipped = b_coeffs.flip(1)`.
   - FIR numerator: `input_signal_windows = torch::nn::functional::conv1d(padded_waveform,
     b_flipped.unsqueeze(1), torch::nn::functional::Conv1dFuncOptions().groups(n_channel))`
     (grouped depthwise conv with the FLIPPED `b`). Result is `(n_batch, n_channel, n_sample)`.
   - Normalize by `a0`: `input_signal_windows.div_(a_coeffs.index({Slice(), Slice(None, 1)}))` and
     `a_flipped = a_flipped / a_coeffs.index({Slice(), Slice(None, 1)})` (use `torch::indexing::Slice`).
   - **Sequential time loop (the only non-vectorizable part):** mirror
     `_lfilter_core_generic_loop`. With `a_flipped3 = a_flipped.unsqueeze(2)` shaped
     `(n_channel, n_order, 1)`, iterate `i_sample` over `0..n_sample-1`:
     `o0 = input_signal_windows.index({Slice(), Slice(), i_sample})` shaped `(n_batch, n_channel)`;
     `windowed_output = padded_output.index({Slice(), Slice(), Slice(i_sample, i_sample + n_order)})`
     shaped `(n_batch, n_channel, n_order)`;
     `o0 -= torch::matmul(windowed_output.transpose(0, 1), a_flipped3).index({"...", 0}).t()`
     (i.e. `bmm` over channels: `(n_channel, n_batch, n_order) @ (n_channel, n_order, 1)` →
     `(n_channel, n_batch, 1)` → `[...,0]` → `.t()` → `(n_batch, n_channel)`); write
     `padded_output.index_put_({Slice(), Slice(), i_sample + n_order - 1}, o0)`.
     (A faster equivalent is a CPU `accessor<...>` double loop, but match this tensor-op form first for
     correctness; optimize only if needed and keep results bit-comparable.)
   - Return `padded_output.index({Slice(), Slice(), Slice(n_order - 1, None)})` (drop the left pad).
3. **`lfilter` wrapper** — mirror torchaudio's `lfilter`:
   - Validate `a_coeffs.sizes() == b_coeffs.sizes()` and `a_coeffs.dim() <= 2` (`> 2` ⇒ raise).
   - If `a_coeffs.dim() > 1`: if `batching`, require `waveform.size(-2) == a_coeffs.size(0)` (else
     raise); else `waveform = torch::stack(std::vector<tensor_t>(a_coeffs.size(0), waveform), -2)`.
     Else (`dim() == 1`): `a_coeffs = a_coeffs.unsqueeze(0)`, `b_coeffs = b_coeffs.unsqueeze(0)`.
   - Pack: `shape = waveform.sizes()`; `packed = waveform.reshape({-1, a_coeffs.size(0), shape.back()})`.
   - `output = _lfilter_core(packed, a_coeffs, b_coeffs)`.
   - `if (clamp) output = torch::clamp(output, -1.0, 1.0);`
   - Unpack: rebuild the original leading dims + the new time length:
     `std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1); out_shape.push_back(output.size(-1));
     return output.reshape(out_shape);` (output time == input time, but keep this general).
4. **`biquad`** — build `a_coeffs = torch::tensor({a0,a1,a2}, waveform.options())`,
   `b_coeffs = torch::tensor({b0,b1,b2}, waveform.options())` (1D, dtype/device from `waveform`); return
   `lfilter(waveform, a_coeffs, b_coeffs, /*clamp=*/true, /*batching=*/true)`. Coeffs are UNNORMALIZED
   (lfilter divides by `a0`). Equivalent to torchaudio's `view(1)+cat`; a single `torch::tensor({...})`
   is fine.
5. **`filtfilt`** — `auto fwd = lfilter(waveform, a_coeffs, b_coeffs, /*clamp=*/false);`
   `auto bwd = lfilter(fwd.flip(-1), a_coeffs, b_coeffs, /*clamp=*/clamp).flip(-1); return bwd;`
   No edge padding.
6. **Add tests + golden + ctest + coverage** (final step):
   - In `main.cpp`, add and register:
     - `test_lfilter_identity_fir`: `b=[1,0,0]`, `a=[1,0,0]` on a 1D/2D signal ⇒ output equals input
       (TM_CHECK_TENSOR_CLOSE, atol 1e-6).
     - `test_lfilter_one_pole`: `a=[1,-0.5]`, `b=[1,0]`, unit impulse input ⇒ closed-form geometric
       response `0.5^n` (TM_CHECK_CLOSE per sample or TM_CHECK_TENSOR_CLOSE vs a built tensor).
     - `test_lfilter_clamp`: large-gain coeffs ⇒ output ∈ [-1,1] when `clamp=true`, can exceed when
       `clamp=false` (assert min/max).
     - `test_lfilter_batching`: 2D coeffs `(C, order)` with a `(C, T)` waveform (batching path) +
       the `batching=false` stack path (shape assertions); plus a raises-test for `a_coeffs.dim()>2`
       and for the `size mismatch` between `a` and `b` (try/catch `std::invalid_argument`,
       `TM_CHECK(threw)`).
     - `test_biquad_analytic`: a known biquad (e.g. b=[0.5,0,0], a=[1,0,0] scaling; and a 2nd-order
       lowpass coeff set) impulse response vs the analytic / `lfilter` reference.
     - `test_filtfilt_zero_phase`: `filtfilt` of a symmetric signal stays symmetric; and equals
       `lfilter(flip(lfilter(x)))` composition.
     - Golden cross-checks `test_lfilter_golden`, `test_biquad_golden`, `test_filtfilt_golden`:
       fixed random coeffs/signal, compare to baked torchaudio outputs (TM_CHECK_TENSOR_CLOSE,
       atol 1e-5, rtol 1e-4).
   - Append a block to `gen_golden.py`: set `torch.manual_seed(0)`, build a fixed `waveform` and
     `a_coeffs`/`b_coeffs`, print `F.lfilter`, `F.biquad`, `F.filtfilt` outputs (shape + a few sample
     values + sum, rounded). Run
     `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`
     and bake the printed constants into `main.cpp` as literals.
   - Build & run:
     `cmake --build build --target audio_test_functional && ./build/unit_test/audio/functional/audio_test_functional`;
     then `ctest --test-dir build` must be green.
   - Confirm 100% line coverage of the new `_functional_filtering.hpp` lines (every branch: 1D vs 2D
     coeffs, batching true/false, clamp true/false, each raise path) via the project coverage run
     (`-DTORCHMEDIA_COVERAGE=ON`, `--ignore-filename-regex='_vendor/.*'`).

## Acceptance criteria
- [ ] `lfilter`, `biquad`, `filtfilt` exist in `_functional_filtering.hpp`, `inline`, namespace
      `torchmedia::audio::functional`, header-only (no new .cpp), with the signatures in Deliverables.
- [ ] `audio.hpp` includes the new header; `torchmedia.hpp` re-exports it; project still builds.
- [ ] Identity-FIR test: output == input within atol 1e-6.
- [ ] One-pole test: impulse response matches closed-form `0.5^n` within atol 1e-6.
- [ ] `clamp=true` outputs ∈ [-1,1]; `clamp=false` path exercised and allowed to exceed.
- [ ] Batching path (2D coeffs) and `batching=false` stack path both shape-correct; `dim()>2` and
      `a/b` size-mismatch raise `std::invalid_argument`.
- [ ] `lfilter` / `biquad` / `filtfilt` match baked torchaudio 2.5.1 golden values within
      atol 1e-5, rtol 1e-4.
- [ ] `ctest --test-dir build` green.
- [ ] 100% line coverage of `_functional_filtering.hpp` (vendored excluded).

## Constraints
- Header-only; `inline` free functions; torch-native ATen ops only
  (`torch::nn::functional::pad`/`conv1d`, `flip`, `zeros_like`, `matmul`/`bmm`, `transpose`/`t`,
  `clamp`, `index`/`index_put_`, `reshape`/`stack`). No system audio/DSP libraries, no compiled op.
- Match torchaudio's validation/raises: equal-size `a`/`b`, coeff `ndim <= 2`, `n_order > 0`, batching
  channel-count check. Use `handle_exceptions<tensor_t, std::invalid_argument>(...)` (throws unless
  `LIB_TORCH_MEDIA_NO_EXCEPTIONS`), matching the existing `convolve` style.
- Coeffs are ordered `[a0, a1, a2, …]`; both `a` and `b` are FLIPPED for the conv/recurrence and
  divided by `a0`. `biquad` passes UNNORMALIZED coeffs (lfilter does the `/a0`).
- `filtfilt` must NOT add scipy-style edge padding (torchaudio doesn't).
- Caveat: the time recurrence is inherently SEQUENTIAL (data dependency across samples) — it is the one
  non-vectorizable piece; keep it correct first. Run it on CPU; do not attempt to vectorize across the
  time axis.
- Keep `clang-format` clean (LLVM base, IndentWidth 4, ColumnLimit 120, namespaces indented).

## Notes / Assumptions
- Assumption: a plain `bool clamp`/`bool batching` argument pair is sufficient — NO `lfilter_option`
  struct is required (torchaudio uses plain kwargs; only add an option struct if Mux asks). If one is
  added later it goes in `_functional_methods_options.hpp` per D2.
- Assumption: input `waveform` may be 1D `(T)` or 2D `(C, T)` or higher; the reshape `(-1, num_filters,
  T)` handles leading dims exactly as torchaudio. For 1D coeffs, `num_filters == 1`. Verify the 1D
  waveform + 1D coeffs path (most common via `biquad`) produces shape `(T)` back out.
- Assumption: float32 is the working dtype (tests use `torch::tensor({...})` default float); coeffs are
  cast to `waveform.options()` in `biquad`. Keep `_lfilter_core` dtype-agnostic.
- Gotcha: do `a_coeffs.div_`/`input_signal_windows.div_` on COPIES, not on the caller's tensors —
  `flip(1)` already returns a new tensor for `a_flipped`/`b_flipped`, but `input_signal_windows` comes
  from `conv1d` (fresh) so in-place is safe there; never mutate the caller's `a_coeffs`/`b_coeffs`.
- Dependency: this task is Tier-0 and UNBLOCKS task02 (biquad designers), task06 (`deemphasis`), and
  task12 (`loudness`). Land it first and keep the `biquad` signature stable for task02.
- Question for Mux: none — `lfilter`/`biquad`/`filtfilt` are firmly IN scope (D3/D6); only the autograd
  backward is deferred (Out), which the project does not need. Confirm at review that a forward-only
  (no-grad) `lfilter` is acceptable (the project ports forward ops only).
