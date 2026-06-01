# Task 16 — Implement rnnt_loss (forward-only, scope-pending)
id: 2026-06-01/task16
parent: 2026-06-01/progress01
status: blocked             # active | blocked | done (Mux decision 2026-06-01)
owner: code_agent

## Objective
Add a torch-native, **forward-cost-only** `rnnt_loss` (RNN-Transducer loss) to a new
`_functional_rnnt.hpp` (or appended to `_functional.hpp`), mirroring torchaudio's wrapper
validation/reduction but computing the cost via an in-C++ alpha forward-backward DP — **no
analytic gradient / no autograd**.

## Scope
In:
- `rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=-1, clamp=-1, reduction="mean",
  fused_log_softmax=true)` — wrapper: validate `reduction in {none,mean,sum}`; remap `blank<0` to
  `logits.shape[-1]+blank`; apply reduction (`mean()` / `sum()` / per-batch vector).
- `rnnt_loss_option` fluent option struct (blank, clamp, reduction, fused_log_softmax) in
  `_functional_methods_options.hpp`.
- The **forward cost** only: per-batch negative log-likelihood from the alpha (forward) DP over the
  `(T x U+1)` transducer lattice in log space.
- Forward-only tests (closed-form tiny lattice + `.venv` torchaudio cross-check for forward cost).

Out:
- **Analytic backward / gradients / `torch::autograd::Function`** — INFEASIBLE header-only; the
  discarded second return value of `torch.ops.torchaudio.rnnt_loss` (the gradient tensor) is NOT
  produced. Output is a plain (non-differentiable) tensor.
- `clamp` gradient-clamping semantics (clamp only affects the backward pass; with no backward it is a
  validated-but-inert parameter here — document it).
- The custom compiled op path (`torch.ops.torchaudio.rnnt_loss`), GPU/`gpu_rnnt`, alphabet/warp-rnnt
  optimizations.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D1 (.venv golden source), D2 (file
   split), this op listed in the "to port" inventory; rationale for scope-pending XL ops.
2. torchaudio v2.5.1 wrapper source —
   `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/functional.py`
   (`rnnt_loss`: signature, `reduction not in [...]` raise, `blank<0` remap, dispatch to
   `torch.ops.torchaudio.rnnt_loss`, reduction `mean`/`sum`/`none`).
3. torchaudio v2.5.1 reference CPU kernel (for the forward DP math, log-sum-exp of emit/blank
   transitions) —
   `https://github.com/pytorch/audio/tree/v2.5.1/src/libtorchaudio/rnnt` (`cpu`/`compute_alphas`,
   `LogSumExp2`).

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional_rnnt.hpp` (NEW) — or append to
  `libtorchmedia/include/torchmedia/_audio/_functional.hpp` (decide per Question below).
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` — add `rnnt_loss_option`.
- `libtorchmedia/include/torchmedia.hpp` — aggregate the new header if a NEW file is created.
- `unit_test/audio/functional/main.cpp` — add + register tests.
- `unit_test/audio/functional/gen_golden.py` — add the golden generators.

## Deliverables
- `_functional_rnnt.hpp` (or `_functional.hpp`): `inline auto rnnt_loss(...) -> torch::Tensor` in
  namespace `torchmedia::audio::functional`, exact signature
  `rnnt_loss(const_tensor_lref_t logits, const_tensor_lref_t targets, const_tensor_lref_t
  logit_lengths, const_tensor_lref_t target_lengths, const rnnt_loss_option &opt = {})`. The
  forward DP returns the per-batch cost vector; the wrapper applies reduction.
- `rnnt_loss_option` in `_functional_methods_options.hpp` with fluent setters
  (`blank(int)`, `clamp(double)`, `reduction(std::string)`, `fused_log_softmax(bool)`), matching the
  existing `spectrogram_option` style (`return *this;`).
- Assertion tests in `main.cpp` registered in `main()`'s call list; new golden constants in
  `gen_golden.py` baked into `main.cpp` (build must NOT depend on `.venv` at runtime).

## Steps
1. **Wrapper validation** — implement signature; validate `reduction in {"none","mean","sum"}` via
   `handle_exceptions<torch::Tensor, std::invalid_argument>(...)` or `TORCH_CHECK` (match
   torchaudio's `ValueError` message). If `blank < 0`, set `blank = logits.size(-1) + blank`
   (default `-1` -> last class). Check `logits` is rank-4 `(B,T,U+1,C)` and the lengths are rank-1
   `(B,)`.
2. **Optional log-softmax** — if `fused_log_softmax`, apply
   `torch::log_softmax(logits, /*dim=*/-1)` to get log-probs `log_probs`; else treat `logits` as
   already-normalized log-probs (torchaudio's `fused_log_softmax=false` path). Use `accessor<float,4>`
   (after `.contiguous()` / `to(kFloat)`) for the sequential DP.
3. **Alpha forward DP per batch** — for each `b`: let `T_b = logit_lengths[b]`, `U_b =
   target_lengths[b]` (target seq length; lattice has `U_b+1` rows). In log space build
   `alpha[t][u]` with `alpha[0][0]=0`; recurrence
   `alpha[t][u] = LogSumExp2( alpha[t-1][u] + blank_logp(t-1,u), alpha[t][u-1] +
   emit_logp(t,u-1) )` where `blank_logp(t,u)=log_probs[b][t][u][blank]` and
   `emit_logp(t,u)=log_probs[b][t][u][targets[b][u]]`. Forward cost
   `= -( alpha[T_b-1][U_b] + blank_logp(T_b-1, U_b) )`. Use a numerically stable
   `log_sum_exp(a,b) = max(a,b) + log1p(exp(-|a-b|))` helper. Store into a `(B,)` cost tensor.
4. **Reduction** — `mean` -> `costs.mean()`, `sum` -> `costs.sum()`, `none` -> return `costs`
   (shape `(B,)`).
5. **Tests + golden + green** — add tests, bake golden via `.venv` gen_golden.py
   (`.venv/bin/python unit_test/audio/functional/gen_golden.py`), keep
   `cmake --build build --target audio_test_functional && ctest --test-dir build` green, and reach
   100% line coverage of the new lines:
   - **Closed-form tiny lattice**: `T=1, U=0` (B=1, logits shape `(1,1,1,C)`) -> cost is exactly
     `-log p(blank)` = `-log_softmax(...)[...,blank]`. Verify with `TM_CHECK_CLOSE`.
   - **torchaudio cross-check** (`gen_golden.py`): small random `(B,T,U+1,C)` (fixed seed) with
     `blank=C-1`; print `F.rnnt_loss(..., reduction=r)` for `r in {none,mean,sum}`; bake the floats;
     compare with `TM_CHECK_CLOSE` / `TM_CHECK_TENSOR_CLOSE`.
   - **Validation tests**: bad `reduction` raises; `blank=-1` remap equals `blank=C-1` result; the
     `fused_log_softmax=false` path (feed pre-`log_softmax`ed logits, expect same cost).

## Acceptance criteria
- [ ] `rnnt_loss` forward cost matches `torchaudio.functional.rnnt_loss` within tolerance
      (`atol<=1e-4`) on the random `(B,T,U+1,C)` golden case for each of `none`/`mean`/`sum`.
- [ ] Closed-form `T=1,U=0` case equals `-log p(blank)` within `1e-5`.
- [ ] Invalid `reduction` raises; `blank<0` remap verified; `fused_log_softmax=false` path verified.
- [ ] `ctest --test-dir build` green; 100% line coverage of the new lines in the target header
      (`_vendor/` excluded).

## Constraints
- Header-only, inline free functions, namespace `torchmedia::audio::functional`; option struct in
  `_functional_methods_options.hpp` with fluent setters.
- Torch-native ATen only (`torch::log_softmax`, `torch::full`/`empty`, `accessor`, `torch::Tensor`
  arithmetic); the DP is an explicit **sequential C++ loop** over `(B,T,U)` (no fused CUDA kernel).
- Match torchaudio's validation/raises exactly (reduction set + message, `blank<0` remap default).
- **No autograd / no gradient**: the result is non-differentiable; do not register a
  `torch::autograd::Function`. Document this caveat at the function and in the progress Agent log.
- XL / scope-pending: if the forward DP proves too costly to verify within the task, downgrade to a
  smoke/forward-only test on a single tiny lattice and mark the task `blocked` pending Mux.

## Notes / Assumptions
- Assumption: `logits` layout is `(B, max_T, max_U+1, C)` per torchaudio; `targets` is `(B, max_U)`
  int; `logit_lengths`/`target_lengths` are `(B,)` int — the DP reads only the valid `T_b`/`U_b`
  prefix.
- Assumption: float32 DP is sufficient for `atol<=1e-4` against torchaudio's CPU kernel; cast inputs
  to `kFloat` and `.contiguous()` before using `accessor`.
- Gotcha: `clamp` affects only the (out-of-scope) backward; here it is validated/stored but inert —
  state this in a comment so it is not mistaken for a no-op bug.
- Gotcha: torchaudio's wrapper unpacks `costs, _ = torch.ops.torchaudio.rnnt_loss(...)` — the second
  value is the gradient we deliberately do not compute.
- Dependency: none on other task headers (self-contained DP); only the shared `handle_exceptions`
  helper (`globel_include.hpp`) and `_functional_methods_options.hpp`.
- **Question for Mux**: (1) Confirm FORWARD-ONLY (non-differentiable cost) is acceptable, or DEFER
  this op entirely until a full differentiable ATen rebuild is scheduled. (2) New file
  `_functional_rnnt.hpp` vs. appending to `_functional.hpp`? (3) Is the inert-`clamp` semantics
  acceptable, or should passing a non-default `clamp` raise "unsupported (forward-only)"?
