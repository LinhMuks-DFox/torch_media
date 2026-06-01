# Task 14 — Implement beamforming (psd, mvdr ×2, rtf ×2, apply_beamforming)
id: 2026-06-01/task14
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Port torchaudio's six multichannel complex beamforming ops — `psd`, `mvdr_weights_souden`,
`mvdr_weights_rtf`, `rtf_evd`, `rtf_power`, `apply_beamforming` — onto native libtorch in a NEW
header `libtorchmedia/include/torchmedia/_audio/_functional_beamforming.hpp`.

## Scope
In:
- `psd(specgram, mask, normalize, eps)` — spatial covariance via einsum + masked time-reduction.
- `apply_beamforming(beamform_weights, specgram)` — einsum projection of weights onto specgram.
- `mvdr_weights_souden(psd_s, psd_n, reference_channel, diagonal_loading, diag_eps, eps)`.
- `mvdr_weights_rtf(rtf, psd_n, reference_channel, diagonal_loading, diag_eps, eps)`.
- `rtf_evd(psd_s)` — relative transfer function via complex Hermitian EVD.
- `rtf_power(psd_s, psd_n, reference_channel, n_iter, diagonal_loading, diag_eps)`.
- Shared inline helpers: `_assert_psd_matrices`, `_compute_mat_trace`, `_tik_reg`.
- The `reference_channel` Union[int, Tensor] split (int overload + one-hot Tensor overload/variant).
- Assertion tests + golden cross-checks for all six.

Out:
- Autograd / backward (forward inference path only; ops are differentiable via ATen but we ship no
  custom `Function`).
- The `MVDR` / `PSD` / `RTFMVDR` / `SoudenMVDR` transform-layer modules (a later progress).
- Any non-complex / real-only fast path; inputs are complex throughout.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D2 (file split: this is the NEW
   `_functional_beamforming.hpp`), D5 (testing rule), D6 (task14 is independent, any order), and the
   Issues note "Beamforming needs complex `linalg::solve`/`eigh`; eigenvectors are phase-ambiguous".
2. `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/functional.py`
   — authoritative source for the six ops + `_assert_psd_matrices` / `_compute_mat_trace` / `_tik_reg`
   (load WebFetch via ToolSearch `select:WebFetch` if you need to confirm signatures/defaults).

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional_beamforming.hpp` (NEW).
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — pattern for `inline auto … -> tensor_t`,
  `handle_exceptions<T, ExceptionT>`, `using tensor_t = torch::Tensor`, `torchmedia::audio::functional`.
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` — if any `xxx_option`
  struct is warranted (only mvdr/rtf_power have enough knobs to justify one; see Notes).
- `libtorchmedia/include/torchmedia/audio.hpp` — add `#include "_audio/_functional_beamforming.hpp"`
  (this is the aggregation point; `torchmedia.hpp` pulls in `audio.hpp`, NOT the leaf header directly).
- `unit_test/audio/functional/main.cpp` — add tests + register them in `main()`.
- `unit_test/audio/functional/gen_golden.py` — extend with the golden generators.

## Deliverables
- `_functional_beamforming.hpp` containing, all `inline` in `namespace torchmedia::audio::functional`:
  - `inline void _assert_psd_matrices(const tensor_t &psd)` — `TORCH_CHECK` complex dtype,
    `dim() >= 2`, and square trailing dims (`size(-1) == size(-2)`). Mirror torchaudio's raises.
  - `inline auto _compute_mat_trace(const tensor_t &input, int64_t dim1 = -2, int64_t dim2 = -1)
    -> tensor_t` — `torch::diagonal(input, 0, dim1, dim2).sum(-1)`.
  - `inline auto _tik_reg(const tensor_t &mat, double reg = 1e-7, double eps = 1e-8) -> tensor_t` —
    `eye = torch::eye(C)`; `scale = trace(mat).real()[..., None, None] * reg`; add
    `eye * scale + eps * eye` on the (complex) diagonal; return `mat + epsilon`.
  - `inline auto psd(const tensor_t &specgram, const tensor_t &mask = {}, bool normalize = true,
    double eps = 1e-10) -> tensor_t`.
  - `inline auto apply_beamforming(const tensor_t &beamform_weights, const tensor_t &specgram)
    -> tensor_t`.
  - `inline auto mvdr_weights_souden(const tensor_t &psd_s, const tensor_t &psd_n,
    int64_t reference_channel, bool diagonal_loading = true, double diag_eps = 1e-7,
    double eps = 1e-8) -> tensor_t` PLUS a one-hot `const tensor_t &reference_channel` overload.
  - `inline auto mvdr_weights_rtf(const tensor_t &rtf, const tensor_t &psd_n,
    /*reference_channel optional*/ …, bool diagonal_loading = true, double diag_eps = 1e-7,
    double eps = 1e-8) -> tensor_t` PLUS a one-hot overload; the int/one-hot ref must be **optional**.
  - `inline auto rtf_evd(const tensor_t &psd_s) -> tensor_t`.
  - `inline auto rtf_power(const tensor_t &psd_s, const tensor_t &psd_n, int64_t reference_channel,
    int64_t n_iter = 3, bool diagonal_loading = true, double diag_eps = 1e-7) -> tensor_t` PLUS a
    one-hot overload.
- (Optional) `mvdr_weights_option` / `rtf_power_option` in `_functional_methods_options.hpp` only if
  the plain-arg signatures get unwieldy — confirm with Mux first (see Question).
- `audio.hpp` updated to include the new header.
- Tests in `main.cpp` (one `static void test_*` per op + helpers) registered in `main()`, plus the
  golden constants baked in from `gen_golden.py`.

## Steps
1. **Scaffold header + helpers** — create `_functional_beamforming.hpp` with the include guard
   `LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_BEAMFORMING_HPP`, `#include "_functional_methods_options.hpp"`
   and `"../globel_include.hpp"`. Implement `_assert_psd_matrices` (`TORCH_CHECK` on
   `input.is_complex()`, `input.dim() >= 2`, `input.size(-1) == input.size(-2)`),
   `_compute_mat_trace` (`torch::diagonal(x, 0, dim1, dim2).sum(-1)`), and `_tik_reg`
   (`torch::eye` scaled by `_compute_mat_trace(mat).real()` unsqueezed to `[..., None, None]` times
   `reg`, plus `eps`, added to `mat`).
2. **psd** — input `(..., channel, freq, time)`. `psd = torch::einsum("...ct,...et->...tce",
   {specgram, specgram.conj()})` -> `(..., freq, time, ch, ch)`. If `mask` defined: when `normalize`,
   `mask = mask / (mask.sum(-1, /*keepdim=*/true) + eps)`; broadcast-multiply `mask.unsqueeze(-1).unsqueeze(-1)`
   (line up to the `(ch,ch)` dims). Sum over time: `.sum(-3)` -> `(..., freq, ch, ch)`. Default
   `eps = 1e-10` (match the signature exactly — NOT 1e-8). No `_assert_psd_matrices` here (output, not input).
3. **apply_beamforming** — weights `(..., freq, channel)`, specgram `(..., channel, freq, time)`;
   return `torch::einsum("...fc,...cft->...ft", {beamform_weights.conj(), specgram})` ->
   `(..., freq, time)`.
4. **mvdr_weights_souden** — `_assert_psd_matrices(psd_s)`/`(psd_n)`; if `diagonal_loading`,
   `psd_n = _tik_reg(psd_n, diag_eps, eps)`; `numerator = torch::linalg::solve(psd_n, psd_s)` (complex);
   `ws = numerator / (_compute_mat_trace(numerator).unsqueeze(-1).unsqueeze(-1) + eps)`; select the
   reference column: int path `beamform_vector = ws.index({..., Slice(), reference_channel})`; one-hot
   Tensor path `torch::einsum("...c,...c->...", {ws, ref})`-style column pick (follow torchaudio's
   `einsum("...c,c->...")` for the one-hot). Return `(..., freq, channel)`.
5. **mvdr_weights_rtf** — `_assert_psd_matrices(psd_n)`; optional `_tik_reg`;
   `numerator = torch::linalg::solve(psd_n, rtf.unsqueeze(-1)).squeeze(-1)`;
   `denominator = torch::einsum("...d,...d->...", {rtf.conj(), numerator})`;
   `beamform_vector = numerator / (denominator.real().unsqueeze(-1) + eps)`; if a reference channel is
   supplied, scale by `rtf[..., ref].conj()` (int) or the one-hot einsum equivalent; if absent, return
   unscaled. Keep the reference **optional**.
6. **rtf_evd** — `_assert_psd_matrices(psd_s)`; `auto [w, v] = torch::linalg::eigh(psd_s)` (Hermitian:
   real ascending eigvals, complex eigvecs); `rtf = v.index({..., Slice(), -1})` (last column = largest
   eigenvalue). Return `(..., freq, channel)`. Eigenvectors are phase-ambiguous — see test step.
7. **rtf_power** — validate `n_iter > 0` (`TORCH_CHECK`); `_assert_psd_matrices(psd_s)`/`(psd_n)`;
   optional `_tik_reg(psd_n)`; `phi = torch::linalg::solve(psd_n, psd_s)` (this counts as iteration 1);
   pick the reference column of `phi` (int or one-hot einsum) and `unsqueeze(-1)` to
   `(..., freq, ch, 1)`. If `n_iter >= 2`: loop `for (i = 0; i < n_iter - 2; ++i) rtf = matmul(phi, rtf);`
   then `rtf = matmul(psd_s, rtf);`. Else (`n_iter == 1`): `rtf = matmul(psd_n, rtf);`. Return
   `rtf.squeeze(-1)` -> `(..., freq, channel)`. **Mirror the iteration count exactly** (the n_iter==1
   vs >=2 branch is load-bearing).
8. **Wire up + tests** — add `#include "_audio/_functional_beamforming.hpp"` to `audio.hpp`. In
   `gen_golden.py` build a small batch of Hermitian PD complex covariance matrices
   (`A = randn(...,C,C, dtype=cfloat); psd = A @ A.conj().transpose(-1,-2)`) for `psd_s`/`psd_n`, a
   complex `rtf`, and a complex `specgram`; print golden outputs (and shapes/sums) for: `psd`
   (mask=None and a normalized mask), `apply_beamforming`, `mvdr_weights_souden` (ref int / one-hot /
   diagonal_loading on+off), `mvdr_weights_rtf` (ref None / int / one-hot), `rtf_power` for
   `n_iter ∈ {1,2,3}`, and for `rtf_evd` print the **phase-invariant outer product**
   `rtf ⊗ rtf.conj()` (NOT raw eigenvectors). Run
   `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`,
   bake the constants into `main.cpp`. Add `static void test_psd/…` using `TM_CHECK`,
   `TM_CHECK_TENSOR_CLOSE`; for `rtf_evd` compare the outer product (phase-invariant) and also a
   libtorch self-reference (explicit-loop EVD vs the op). Add an int↔one-hot equivalence assertion and a
   `TORCH_CHECK`-raises test (`n_iter <= 0`, non-square / non-complex psd). Register every new
   `test_*` in `main()`'s call list.
9. **Green + coverage** — `cmake --build build --target audio_test_functional &&
   ./build/unit_test/audio/functional/audio_test_functional`; `ctest --test-dir build` green; confirm
   100% line coverage of the new lines in `_functional_beamforming.hpp` (every branch: mask on/off,
   normalize on/off, diagonal_loading on/off, int vs one-hot ref, n_iter 1 vs >=2, each `TORCH_CHECK`).

## Acceptance criteria
- [ ] `psd`, `apply_beamforming`, `mvdr_weights_souden`, `mvdr_weights_rtf`, `rtf_evd`, `rtf_power`
      exist `inline` in `_functional_beamforming.hpp` (namespace `torchmedia::audio::functional`) with
      the snake_case signatures above (int + one-hot ref variants where applicable).
- [ ] Each op matches torchaudio 2.5.1 within tolerance on the golden cases: `psd` (mask None +
      normalized mask), `apply_beamforming`, `mvdr_weights_souden` (int/one-hot, loading on/off),
      `mvdr_weights_rtf` (None/int/one-hot), `rtf_power` (n_iter 1/2/3).
- [ ] `rtf_evd` matches torchaudio on the phase-invariant outer product `rtf ⊗ rtf.conj()` (raw
      eigenvectors NOT compared).
- [ ] int-ref and one-hot-ref results are asserted equal; invalid input (`n_iter <= 0`, non-square or
      non-complex psd) raises via `TORCH_CHECK`.
- [ ] `audio.hpp` includes the new header; `ctest --test-dir build` green; 100% line coverage of the
      new lines (vendored `_vendor/` excluded).

## Constraints
- Header-only: all functions `inline` in the new `.hpp`; no `.cpp`, no new runtime deps.
- Torch-native only: `torch::einsum`, `torch::diagonal`, `torch::eye`, `torch::real`/`.real()`,
  `torch::matmul`, `torch::linalg::solve`, `torch::linalg::eigh` (LAPACK-backed, header-only OK),
  `.conj()`, `.unsqueeze`, `torch::indexing::Slice`. No custom compiled ops.
- COMPLEX throughout: `linalg::solve`/`eigh` operate on `complex64`/`complex128`; keep dtypes consistent
  with the input (do not silently downcast). `eigh` returns real eigenvalues, complex eigenvectors,
  ascending order — the last column is the principal eigenvector.
- Match torchaudio's validation/raises (`_assert_psd_matrices`, `n_iter > 0`). Use `TORCH_CHECK` (or
  the project `handle_exceptions<T, ExceptionT>` helper) and mirror the messages where reasonable.
- Mirror torchaudio's `rtf_power` iteration count exactly — the `n_iter == 1` path multiplies by
  `psd_n`, the `>= 2` path loops `n_iter - 2` then multiplies by `psd_s`.
- `psd` default `eps` is `1e-10` (signature), distinct from the mvdr `eps = 1e-8` / `diag_eps = 1e-7`.

## Notes / Assumptions
- Assumption: task14 is independent (progress01 D6) — no dependency on task01 (`lfilter`) or any other
  task. It can be implemented in isolation.
- Assumption: `reference_channel` is modeled as an `int64_t` overload PLUS a `const tensor_t &`
  (one-hot) overload per op; the int path is the common case, the one-hot path mirrors torchaudio's
  `Tensor` branch. `mvdr_weights_rtf` makes the reference **optional** (a no-ref overload that returns
  the unscaled vector).
- Assumption: the `.venv` (torch/torchaudio 2.5.1) is the golden source at dev time only; baked
  constants must not make the build depend on it (CLAUDE.md / progress01 gotcha).
- Gotcha: `rtf_evd` eigenvectors are phase-ambiguous (`eigh` may return `v` or `e^{iθ}·v`). Test the
  outer product `rtf @ rtf.conj().transpose(-1,-2)` (or elementwise `rtf ⊗ rtf.conj()`), which is
  phase-invariant — never assert on raw eigenvector entries.
- Gotcha: `_tik_reg` adds regularization to the **real** part of the diagonal only (`trace.real() * reg
  + eps`), then adds back as a complex eye; confirm the diagonal is updated on the complex matrix, not a
  real view.
- Gotcha: einsum subscripts are exact and order-sensitive — `psd` is `"...ct,...et->...tce"`,
  `apply_beamforming` is `"...fc,...cft->...ft"`; transposing a subscript silently changes the result.
- Question for Mux: should mvdr/rtf_power expose an `xxx_option` struct (fluent setters, matching the
  `spectrogram_option` convention) or keep the flat positional args? The flat signatures are short
  enough that I lean toward NO option struct unless you prefer the uniform option-struct style.
