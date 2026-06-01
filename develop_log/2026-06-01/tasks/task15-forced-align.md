# Task 15 — Forced alignment: forced_align, merge_tokens, TokenSpan
id: 2026-06-01/task15
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Reimplement torchaudio's CTC forced alignment — `forced_align`, `merge_tokens`, and the
`token_span` value type — as header-only, torch-native C++ in the NEW header
`libtorchmedia/include/torchmedia/_audio/_functional_alignment.hpp`.

## Scope
In:
- `struct token_span` — fields `int token; int start; int end; double score;` plus
  `int length() const { return end - start; }` (mirror of torchaudio `TokenSpan.__len__`).
- `forced_align(log_probs, targets, input_lengths={}, target_lengths={}, blank=0)
  -> std::tuple<torch::Tensor, torch::Tensor>` — B==1 Viterbi DP over the CTC trellis.
- `merge_tokens(tokens, scores, blank=0) -> std::vector<token_span>` — collapse repeat runs,
  drop blank runs, mean per-run score.
- Assertion tests in `unit_test/audio/functional/main.cpp` + golden constants in `gen_golden.py`.

Out:
- Batch > 1 (torchaudio v2.5.1 itself is B==1 only; raise on B>1).
- Backward / autograd (forced_align is a no-grad alignment op).
- GPU-specific kernels and the compiled `torchaudio::forced_align` op path (we hand-write the DP).
- Any `input_lengths`/`target_lengths` dtype beyond int (accept int32/int64).

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D2 (file split → this NEW header),
   D4 (forced_align is **scope-pending, recommend INCLUDE**), D5 (testing/coverage rule).
2. `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/_alignment.py`
   — authoritative signatures, validation/raises, and `merge_tokens` boundary logic / `TokenSpan`.
3. `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — existing conventions:
   `namespace torchmedia::audio::functional`, `inline auto … -> tensor_t`, `using namespace
   torch::indexing`, `tensor_t` alias.
4. `libtorchmedia/include/torchmedia/globel_include.hpp:25,33` — `handle_exceptions<T,ExceptionT>(ret,
   msg)` helper (validation/raises). `TORCH_CHECK` is also acceptable.

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional_alignment.hpp` (NEW — create).
- `libtorchmedia/include/torchmedia.hpp` (aggregate: add `#include` for the new header).
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp`
  (only if you choose to add a `forced_align_option`; otherwise plain args — see Notes).
- `unit_test/audio/functional/main.cpp` (add tests + register in `main()`).
- `unit_test/audio/functional/gen_golden.py` (emit golden path/scores when the extension is present).

## Deliverables
- `_functional_alignment.hpp` with, in `namespace torchmedia::audio::functional`:
  - `struct token_span { int token; int start; int end; double score; int length() const; };`
  - `inline auto forced_align(const tensor_t &log_probs, const tensor_t &targets,
    const tensor_t &input_lengths = {}, const tensor_t &target_lengths = {}, int64_t blank = 0)
    -> std::tuple<tensor_t, tensor_t>;`
  - `inline auto merge_tokens(const tensor_t &tokens, const tensor_t &scores, int64_t blank = 0)
    -> std::vector<token_span>;`
- `torchmedia.hpp` includes the new header alongside the other `_audio/_functional*` headers.
- Tests `test_forced_align_*`, `test_merge_tokens_*`, `test_token_span_smoke` in `main.cpp`,
  registered in `main()`'s call list (target ctest: `audio_test_functional`).
- New golden constants in `gen_golden.py` (printed) baked into `main.cpp` (no runtime `.venv` dep).

## Steps
1. **Create header + namespace skeleton** — `#pragma once`, include guard
   `LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_ALIGNMENT_HPP`, include `<ATen/core/TensorBody.h>`,
   `<torch/torch.h>`, `"../globel_include.hpp"`. Open `namespace torchmedia::audio::functional`.
   Define `struct token_span` (POD + `length()`).
2. **`forced_align` validation** (match torchaudio raises) —
   - require `log_probs.dim()==3` (B,T,C) and `targets.dim()==2` (B,L); `B==1` (else raise
     "forced_align supports batch_size == 1 only").
   - default `input_lengths` → full `T`, `target_lengths` → full `L` when the tensor is undefined
     (`!t.defined()` / `t.numel()==0`).
   - `blank` in `[0, C)`; raise if `targets` contains `blank`; raise if `max(targets) >= C`.
   - `T >= L + N_repeat` where `N_repeat` = count of `i` with `targets[i]==targets[i-1]` (each
     repeat needs an interleaving blank). Raise "targets length is too long for CTC".
3. **Build the expanded target trellis** — for the single batch row take `L = target_lengths[0]`,
   `T = input_lengths[0]`. Expanded state sequence `S = 2L+1`: `state[2k]=blank`, `state[2k+1]=
   targets[k]`. Allocate `alphas` `torch::full({T, S}, -inf)` and an int `backptr` `torch::zeros({T,
   S}, kInt)` (backpointer ∈ {0=stay i, 1=advance i-1, 2=skip-blank i-2}).
4. **Viterbi forward (SEQUENTIAL time loop)** — get `auto lp = log_probs.accessor<float,3>();`,
   `alphas.accessor<float,2>()`, `backptr.accessor<int,2>()`. Init `t=0`: `alphas[0][0]=lp[0][0]
   [blank]`, `alphas[0][1]=lp[0][0][state[1]]`. For `t=1..T-1`, for each reachable state `s`:
   candidate from stay `s`, advance `s-1`, and skip `s-2` **only when** `state[s]` is a non-blank
   that differs from `state[s-2]` (i.e. `s>=2 && state[s]!=blank && state[s]!=state[s-2]`); take the
   max, store the argmax in `backptr`, add `lp[0][t][state[s]]`. (This is the one non-vectorizable
   primitive — a hand-written loop, like `lfilter` in task01.)
5. **Backtrace** — pick the better of the two terminal states `S-1` (last blank) and `S-2` (last
   label) at `t=T-1`; walk `backptr` back to `t=0` recording, per time step, the chosen `state`'s
   token id and its emission log-prob. Reverse to time order.
6. **Outputs** — `paths` `torch::tensor` shape `(B=1, T)` int64 of the per-step token index;
   `scores` shape `(B=1, T)` float of the chosen-label log-prob at each step. Return
   `std::make_tuple(paths, scores)`. (torchaudio returns `(paths, scores)` of shape `(B,T)`.)
7. **`merge_tokens`** — validate `tokens.dim()==1 && scores.dim()==1` and `numel()` equal (raise
   "tokens and scores must be 1-D, equal length"). Find run boundaries where the token value
   changes: use a torch `diff` with a prepended/appended sentinel `-1`
   (`torch::cat` then `torch::nonzero`), OR a plain C++ boundary scan over an
   `accessor<int64_t,1>` (either is fine — the scan is simpler and avoids the scalar/single-change
   edge). For each run `[start,end)`: if `token != blank`, push `token_span{token, start, end,
   mean(scores[start:end])}`. Blank runs are dropped; consecutive equal non-blank tokens collapse to
   one span. Handle the degenerate single-element / single-change case (one run spanning all T).
8. **Aggregate + tests + golden** — add the include to `torchmedia.hpp`. Add to `main.cpp`:
   - `test_token_span_smoke` — `token_span{5,3,7,0.5}; TM_CHECK(s.length()==4);`.
   - `test_forced_align_closed_form` — a tiny hand-built `log_probs` (e.g. T small, C=3) where the
     optimal monotonic path is derivable by hand; assert `paths` and per-step `scores` exactly
     (`TM_CHECK`, `TM_CHECK_TENSOR_CLOSE`).
   - `test_forced_align_validation` — assert raises on B>1, blank-in-targets, `max>=C`,
     `T < L+N_repeat` (wrap in try/catch like existing `test_convolve_broadcast_and_errors`).
   - `test_merge_tokens_closed_form` — `tokens=[0,0,1,1,1,0,2,2]`, `scores` chosen so means are
     exact; assert two spans: `{token=1,start=2,end=5,score=mean}` and `{token=2,start=6,end=8,
     score=mean}`; assert `length()` of each.
   - `test_forced_align_vs_torchaudio` — point-wise vs `torchaudio.functional.forced_align`
     golden (B=1) baked from `gen_golden.py` **only when the extension is available** (the upstream
     op is a C++ extension; if `.venv` import fails, fall back to the closed-form case — see Notes).
   - Register every new `test_*` in `main()`'s call list (before `return tm_test::summary(...)`).
   - Extend `gen_golden.py`: build the same small `log_probs`/`targets`, call
     `F.forced_align(...)`, print `paths`/`scores` lists; print `merge_tokens` span tuples. Run
     `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`
     and bake the printed constants into `main.cpp`. Then `cmake --build build --target
     audio_test_functional && ./build/unit_test/audio/functional/audio_test_functional`;
     `ctest --test-dir build` green; confirm **100% line coverage** of the new lines in
     `_functional_alignment.hpp`.

## Acceptance criteria
- [ ] `forced_align` on the hand-built case returns the exact closed-form `paths` and per-step
      `scores`; both are shape `(1,T)` (int64 paths, float scores).
- [ ] `forced_align` matches `torchaudio.functional.forced_align` golden (B=1) within
      `TM_CHECK_TENSOR_CLOSE(..., 1e-5, 1e-5)` for `scores` and exact for `paths` (when the
      extension is available; otherwise the closed-form case carries coverage).
- [ ] All four validation raises fire (B>1, blank∈targets, max≥C, T too short).
- [ ] `merge_tokens([0,0,1,1,1,0,2,2], …)` yields exactly the two expected non-blank spans with
      correct `start`/`end`/mean `score`; `length()==end-start`.
- [ ] `token_span` smoke: `length()` returns `end - start`.
- [ ] `ctest --test-dir build` green; **100% line coverage** of `_functional_alignment.hpp`
      (vendored `_vendor/` excluded; build/coverage do not depend on `.venv` at runtime).

## Constraints
- **Header-only**: all logic `inline` in `_functional_alignment.hpp`, namespace
  `torchmedia::audio::functional`; `snake_case`.
- **Torch-native only** for buffers/tensor ops: `torch::full`/`torch::zeros`/`torch::empty`,
  `torch::tensor`, `accessor<float,3>` / `accessor<float,2>` / `accessor<int,2>`,
  `torch::cat`/`torch::nonzero`/`torch::diff` where used. The Viterbi time loop is an explicit
  C++ loop over time (each step depends on the previous) — this is the accepted non-vectorizable
  primitive, same caveat as `lfilter`.
- **Match torchaudio validation/raises** exactly (use `handle_exceptions<...>` or `TORCH_CHECK`);
  B==1 only, as upstream.
- **Scope-pending (D4)**: this task is recommended-INCLUDE but Mux confirms at review — see Question.

## Notes / Assumptions
- Assumption: `tensor_t` and `handle_exceptions<T,ExceptionT>` are already visible via
  `globel_include.hpp` (as used throughout `_functional.hpp`); no new aliases needed.
- Assumption: torchaudio's `forced_align` `scores` are the **emission log-prob of the chosen label
  at each time step** (`log_softmax`-space if the caller pre-applied it). We do **not** apply
  `log_softmax` inside the op — the input is already log-probabilities (mirror upstream, which takes
  CTC emissions as `log_probs`). State this in a header comment.
- Assumption: prefer **plain args** over a `forced_align_option` struct (the params are few and
  positional-clear); only add `forced_align_option` to `_functional_methods_options.hpp` if Mux
  prefers the fluent-setter convention for symmetry — flag it but default to plain args.
- Gotcha: the skip transition (`s-2`) is allowed only between **distinct** non-blank labels; allowing
  it across equal labels would illegally merge a required interleaving blank — this is the source of
  the `N_repeat` length requirement, keep them consistent.
- Gotcha: `merge_tokens` single-change / single-element case — the boundary scan must still emit the
  final run; if you use the `diff`+`nonzero` form, prepend AND append the `-1` sentinel so the last
  run closes (the scalar/0-d `nonzero` result is the trap the FUNCTION DATA warns about). The plain
  C++ scan sidesteps this; either is acceptable.
- Dependency: **independent** of task01 (`lfilter`) and task14 (beamforming) — D6 lists task15 as
  any-order. No upstream task blocks it.
- Question for Mux: confirm INCLUDE of `forced_align` for this milestone (D4 scope-pending). If the
  `.venv` torchaudio CTC extension is unavailable on this machine, is the **closed-form +
  validation** coverage sufficient as the acceptance gate (deferring the point-wise golden), or
  should the cross-check block task completion?
