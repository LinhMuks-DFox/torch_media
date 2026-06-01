# Task 13 — Implement metrics: edit_distance, frechet_distance
id: 2026-06-01/task13
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Add the two `torchaudio.functional` metrics — `edit_distance` (Levenshtein over arbitrary
sequences) and `frechet_distance` (Fréchet distance between two Gaussians) — as inline functions
appended to `_functional.hpp`, each with assertion + `.venv`-golden tests.

## Scope
In:
- `edit_distance` — templated, non-tensor: `template<class Seq> inline int64_t edit_distance(const Seq& seq1, const Seq& seq2)`.
- `frechet_distance` — tensor: `frechet_distance(mu_x, sigma_x, mu_y, sigma_y) -> torch::Tensor` (scalar).
Out:
- No autograd/backward (forward value only; both are pure metrics, no grad path in torchaudio).
- No `frechet_distance_option` struct — neither function takes tunable options.
- No batching beyond what torchaudio supports (`frechet_distance` is single-pair; `mu` 1-D, `sigma` 2-D).
- Compressed codecs / unrelated ops (other tasks).

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D1 (golden via `.venv`), D5 (testing/coverage), D6 (task13 is independent / any order).
2. torchaudio v2.5.1 source — `edit_distance` + `frechet_distance` reference:
   https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/functional.py
   (fetch via ToolSearch `select:WebFetch` if you need to confirm validation messages / arg order).

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — APPEND both functions inside
  `namespace torchmedia::audio::functional` (before the closing `}` after `resample`); reuse the
  existing `handle_exceptions<T, ExceptionT>(...)` helper / `TORCH_CHECK` for validation.
- `unit_test/audio/functional/main.cpp` — add `test_edit_distance()` + `test_frechet_distance()`,
  register both in `main()`'s call list (before `return tm_test::summary(...)`).
- `unit_test/audio/functional/gen_golden.py` — extend to emit the `frechet_distance` golden scalar(s).
- (No new option struct, so `_functional_methods_options.hpp` is untouched.)

## Deliverables
- `_functional.hpp`: two inline functions.
  - `template<class Seq> inline int64_t edit_distance(const Seq& seq1, const Seq& seq2)` — two-row DP,
    no tensors, works for `std::string`, `std::vector<int64_t>`, `std::vector<std::string>`, etc.
  - `inline auto frechet_distance(const_tensor_lref_t mu_x, const_tensor_lref_t sigma_x, const_tensor_lref_t mu_y, const_tensor_lref_t sigma_y) -> tensor_t`
    (use the project's `const_tensor_lref_t` / `tensor_t` aliases as the surrounding functions do).
- `main.cpp`: `test_edit_distance()`, `test_frechet_distance()` with assertion macros, registered in `main()`.
- `gen_golden.py`: appended block printing the `frechet_distance` golden value(s) for the diagonal-covariance case.

## Steps
1. **edit_distance (two-row Levenshtein)** — append the templated free function. Let
   `m = seq1.size()`, `n = seq2.size()`. Use two `std::vector<int64_t>` rows: `dold` initialized to
   `0,1,...,n` (`std::iota`); `dnew` of size `n+1`. For `i` in `1..m`: `dnew[0]=i`; for `j` in `1..n`:
   if `seq1[i-1]==seq2[j-1]` then `dnew[j]=dold[j-1]` (carry, cost 0) else
   `dnew[j]=std::min({dold[j-1]+1, dnew[j-1]+1, dold[j]+1})` (sub / ins / del); then `std::swap(dold,dnew)`.
   Return `dold[n]`. Handle empty sequences naturally (loop bounds make `edit_distance(empty,x)=len(x)`).
   No tensor / torch dependency; uses `<vector>`, `<numeric>` (already included), `<algorithm>` for `std::min`.
2. **frechet_distance (Gaussian Fréchet / FAD)** — append the tensor function. Validate, then compute:
   - Validation (mirror torchaudio raises; use `TORCH_CHECK` / `handle_exceptions`):
     `mu_x.dim()==1` and `mu_y.dim()==1` ("mu_x/mu_y must be 1-D");
     `sigma_x.dim()==2 && sigma_x.size(0)==sigma_x.size(1) && sigma_x.size(0)==mu_x.size(0)` (square, matches mu_x);
     same for `sigma_y` vs `mu_y`; `mu_x.size(0)==mu_y.size(0)` (shapes match).
   - `a = (mu_x - mu_y).square().sum();`
   - `b = sigma_x.trace() + sigma_y.trace();` (use `torch::Tensor::trace`).
   - `c = torch::linalg::eigvals(torch::matmul(sigma_x, sigma_y)).sqrt().real().sum();`
     NOTE: `eigvals` returns **complex** eigenvalues of the (generally non-symmetric) product
     `sigma_x @ sigma_y`; take `.sqrt()` on the complex tensor first, then `.real()`, then `.sum()`.
   - `return a + b - 2 * c;` (scalar tensor; same dtype as inputs).
3. **Tests** — in `main.cpp`:
   - `test_edit_distance()`: closed-form — `edit_distance(std::string("abc"), std::string("abc")) == 0`;
     single substitution `("abc","abd") == 1`; single insertion `("abc","abxc") == 1`;
     the canonical `("kitten","sitting") == 3`; empty cases `("","abc") == 3`, `("abc","") == 3`.
     Also exercise a non-string `Seq`: `std::vector<int64_t>{1,2,3}` vs `{1,2,4}` -> 1 (covers the template).
   - `test_frechet_distance()`:
     - identical distributions -> `0`: `mu_x==mu_y`, `sigma_x==sigma_y` (e.g. `torch::eye(3)`) gives
       `frechet_distance == 0` within tol (TM_CHECK_CLOSE, atol ~1e-4).
     - diagonal-covariance analytic case: with diagonal `sigma_x=diag(sx)`, `sigma_y=diag(sy)` and
       means `mu_x,mu_y`, the matrix sqrt is closed-form, so
       `fd = sum((mu_x-mu_y)^2) + sum(sx) + sum(sy) - 2*sum(sqrt(sx*sy))`. Assert against this hand value.
     - libtorch self-reference + `.venv` cross-check: bake the `frechet_distance` golden scalar printed by
       `gen_golden.py` for one general (non-diagonal SPD) covariance pair and assert with TM_CHECK_CLOSE
       (loosen atol/rtol slightly — watch `eigvals` ordering/precision differences between ATen and numpy).
     - Validation: a wrong-shape input (e.g. 2-D `mu_x`, or `sigma_x` non-square) raises (TM_CHECK on a
       caught exception, matching how `test_convolve_broadcast_and_errors` checks raises).
4. **Golden + green** — extend `gen_golden.py` to construct the same general SPD covariance pair
   (use `torch.manual_seed(...)`, `A@A.T + eps*I` for SPD) and `mu` vectors, call
   `torchaudio.functional.frechet_distance`, and `print` the scalar; bake the constant into `main.cpp`
   (build must NOT depend on `.venv`). Run the generator with
   `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`.
   Then build & run:
   `cmake --build build --target audio_test_functional && ./build/unit_test/audio/functional/audio_test_functional`;
   `ctest --test-dir build` must be green; confirm 100% line coverage of the new lines in `_functional.hpp`
   (all `edit_distance` branches: match vs the 3-way min; both `frechet_distance` validation raises + the
   happy path) via the coverage run (`-DTORCHMEDIA_COVERAGE=ON`, ignoring `_vendor/.*`).

## Acceptance criteria
- [ ] `edit_distance` returns `0` for equal sequences, `1` for a single edit, `3` for `kitten`/`sitting`,
      and the right length for empty-vs-nonempty — for both `std::string` and a numeric-vector `Seq`.
- [ ] `frechet_distance` returns `0` (within tol) for identical distributions and matches the closed-form
      diagonal-covariance value, and matches the `.venv` torchaudio golden for the general SPD case within tolerance.
- [ ] Wrong-shape inputs to `frechet_distance` raise (matching torchaudio's validation).
- [ ] Both test functions are registered in `main()`; `ctest --test-dir build` is green.
- [ ] 100% line coverage of the new lines in `_functional.hpp` (every `edit_distance` branch + both
      `frechet_distance` validation paths + happy path).

## Constraints
- Header-only: both functions are `inline` (the templated one is implicitly inline) inside
  `namespace torchmedia::audio::functional` in `_functional.hpp`. No new TU, no system deps.
- `frechet_distance` uses torch-native ATen ops only: `torch::matmul`, `torch::linalg::eigvals`,
  `Tensor::sqrt/real/sum/trace/square`. `edit_distance` is plain C++ (std containers), no torch.
- Match torchaudio's validation/raises (mu 1-D; sigma 2-D square matching mu; mu shapes equal) — use
  `TORCH_CHECK` or `handle_exceptions<T, ExceptionT>(...)` as the rest of the header does.
- COMPLEX-linalg caveat: `eigvals(sigma_x @ sigma_y)` is complex; `.sqrt()` MUST run on the complex
  tensor before `.real()` (sqrt of a negative real eigenvalue is imaginary — taking `.real()` first
  would silently lose it). Eigenvalue ordering/precision differs from numpy → use a tolerance on the
  cross-check, not exact equality.
- `gen_golden.py` / `.venv` are dev-only; baked constants must keep the build independent of `.venv`.

## Notes / Assumptions
- Assumption: task13 is independent — it depends on NO other task (no `lfilter`, no options struct);
  it can be done in any order per D6.
- Assumption: `const_tensor_lref_t` / `tensor_t` / `tensor_options_t` aliases and `handle_exceptions`
  are already visible in `_functional.hpp` (used by every existing function there); reuse them.
- Assumption: `frechet_distance` operates on a single (mu, sigma) pair per distribution — no leading
  batch dim (matches torchaudio v2.5.1).
- Gotcha: `edit_distance` cost recurrence is sub `dold[j-1]+1`, ins `dnew[j-1]+1`, del `dold[j]+1`;
  the equal-char branch carries `dold[j-1]` with cost 0 (do NOT add 1). Off-by-one here is the only
  way this function fails.
- Gotcha: keep `frechet_distance` dtype-agnostic — return the tensor result (don't force `.item()`),
  so the test can `.item<double>()` it; feed float64 inputs in the general-SPD test for tighter
  agreement with numpy/torchaudio.
