# Task 11 — Sequential feature ops: sliding_window_cmn, detect_pitch_frequency
id: 2026-06-01/task11
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Append torch-native `sliding_window_cmn` and `detect_pitch_frequency` (+ their NCCF/median helpers)
to `_functional.hpp`, faithfully transcribing torchaudio v2.5.1's inherently sequential per-frame /
per-lag loops, each with `.venv` golden regression tests and 100% line coverage.

## Scope
In:
- `sliding_window_cmn(specgram, sliding_window_cmn_option = {})` — running-mean (and optional
  running-variance) cepstral mean normalization with the centered/left window + `min_cmn_window`
  latency rule and end clamp.
- `detect_pitch_frequency(waveform, sample_rate, detect_pitch_option = {})` — NCCF + per-frame max
  + median-smoothing pitch tracker.
- Private helpers `_compute_nccf`, `_find_max_per_frame`, `_combine_max`, `_median_smoothing`
  (inline, exposed enough to be unit-testable against torchaudio internals).
- `sliding_window_cmn_option` and `detect_pitch_option` option structs (fluent setters).
- Assertion tests + any new golden constants.

Out:
- Autograd / backward (forward-only; these are non-differentiable index ops anyway).
- Any transform-layer wrapper (`SlidingWindowCmn` / `PitchShift` transforms) — later progress.
- Batched-GPU optimization; a straightforward C++ host loop over frames/lags is acceptable.
- No new compressed-codec or FFmpeg paths.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — parent progress; D1 (.venv golden),
   D5 (mandatory tests / coverage), and the L-difficulty "sequential loop" classification for both.
2. torchaudio v2.5.1 source (authoritative algorithm + default args):
   `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/functional.py`
   — functions `sliding_window_cmn`, `detect_pitch_frequency`, `_compute_nccf`,
   `_find_max_per_frame`, `_combine_max`, `_median_smoothing`.

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — APPEND the two ops + 4 helpers here.
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` — add the two option
  structs (mirror the `spectrogram_option` fluent-setter style).
- `unit_test/audio/functional/main.cpp` — add tests + register them in `main()`.
- `unit_test/audio/functional/gen_golden.py` — extend to emit golden constants.

## Deliverables
- In `_functional.hpp` (namespace `torchmedia::audio::functional`, all `inline`):
  - `inline auto sliding_window_cmn(tensor_t specgram, const sliding_window_cmn_option &opt = {}) -> tensor_t;`
  - `inline auto detect_pitch_frequency(const_tensor_lref_t waveform, int sample_rate, const detect_pitch_option &opt = {}) -> tensor_t;`
  - `inline auto _compute_nccf(const_tensor_lref_t waveform, int sample_rate, double frame_time, int freq_low) -> tensor_t;`
  - `inline auto _find_max_per_frame(const_tensor_lref_t nccf, int sample_rate, int freq_high) -> tensor_t;`
  - `inline auto _combine_max(std::pair<tensor_t,tensor_t> a, std::pair<tensor_t,tensor_t> b, double thresh = 0.99) -> std::pair<tensor_t,tensor_t>;`
  - `inline auto _median_smoothing(const_tensor_lref_t indices, int win_length) -> tensor_t;`
- In `_functional_methods_options.hpp`:
  - `sliding_window_cmn_option` with fields `int _cmn_window = 600; int _min_cmn_window = 100; bool _center = false; bool _norm_vars = false;` and fluent setters `cmn_window/min_cmn_window/center/norm_vars`.
  - `detect_pitch_option` with `double _frame_time = 1e-2; int _win_length = 30; int _freq_low = 85; int _freq_high = 3400;` and fluent setters.
- In `main.cpp`: `test_sliding_window_cmn*`, `test_detect_pitch_frequency`, `test_nccf_median_helpers`,
  registered in `main()`; plus a `test_*_option_setters` if needed to hit setter lines.
- New golden constants in `gen_golden.py`, baked as literals into `main.cpp`.

## Steps
1. **Options** — add both option structs to `_functional_methods_options.hpp` with the exact
   torchaudio defaults (`cmn_window=600, min_cmn_window=100, center=false, norm_vars=false`;
   `frame_time=1e-2, win_length=30, freq_low=85, freq_high=3400`) and fluent setters returning `*this`.
2. **sliding_window_cmn** — view input as `(channels, num_frames, num_feats)`: record `was_2d` when
   `specgram.dim()==2`, reshape leading dims into one `channels` axis. Precompute
   `cumsum = x.cumsum(1)` (and `cumsumsq = (x*x).cumsum(1)` if `norm_vars`). Loop `t` in
   `0..num_frames`:
   - `window_start = center ? t - cmn_window/2 : t - cmn_window;`
     `window_end = window_start + cmn_window;`
   - clamp: `if (window_start < 0) { window_end -= window_start; window_start = 0; }`
     `if (window_end > num_frames) { window_start -= (window_end - num_frames); window_end = num_frames; if (window_start < 0) window_start = 0; }`
   - non-center latency rule: `if (!center && window_end > t) window_end = std::max(t + 1, min_cmn_window);`
   - maintain `cur_sum` (and `cur_sumsq`) incrementally vs `last_window_start/last_window_end`: on the
     FIRST iteration seed `cur_sum = cumsum[:, window_end-1, :]` (use `cumsumsq[:, window_end-1, :]`
     for sumsq) — i.e. the "cumsum[:,-1,:]" first-iter path; thereafter add the newly entered frames
     (`x[:, last_window_end .. window_end)`) and subtract the dropped frames
     (`x[:, last_window_start .. window_start)`) so the work per frame is O(delta), not O(window).
   - let `W = window_end - window_start;` set `cmn[:, t, :] = x[:, t, :] - cur_sum / W;`
   - if `norm_vars`: `variance = cur_sumsq / W - (cur_sum*cur_sum)/(W*W);`
     when `W == 1` the variance is degenerate — zero those single-frame windows (set the scale to 0,
     matching torchaudio) else multiply by `variance.clamp_min(eps).pow(-0.5)`.
   - restore the original leading shape; `squeeze(0)` the channels axis when `was_2d`.
   ATen ops: `cumsum`, `slice`/`select` for frame views, `pow`, `clamp_min`, in-place `index_put_`
   or `narrow` slice-assign for `cmn[:, t, :]`.
3. **_compute_nccf** — `EPSILON = 1e-9`. `frame_size = ceil(sample_rate*frame_time)`;
   `lags = ceil(sample_rate/freq_low)` (max lag). Pack waveform to `(-1, time)`. Compute
   `num_of_frames` and pad amount `p = lags + num_of_frames*frame_size - waveform_length`, right-pad.
   For `lag` in `1..lags`: build the two shifted frame views via `unfold(-1, frame_size, frame_size)`
   (`s1` = base frames, `s2` = frames shifted by `lag`), compute
   `phi = (s1*s2).sum(-1) / (EPSILON + linalg::vector_norm(s1, /*ord=*/2, -1)).pow(2) / (EPSILON + linalg::vector_norm(s2, 2, -1)).pow(2)`;
   collect per-lag `phi` and stack to `(..., num_frames, lags)` (lag on the last axis).
4. **_find_max_per_frame** — `lag_min = ceil(sample_rate/freq_high)`. Take `best = max(nccf[..., lag_min:], dim=-1)`
   (values+indices) and `half = max(nccf[..., lag_min : (nccf.size(-1)+lag_min)/2 ...], dim=-1)` over the
   first-half lags; `_combine_max(half, best)` prefers the earlier-half lag when its value
   `> 0.99 * best_value`. Then `indices += lag_min + 1` — the `+1` calibration offset is mandatory and
   a known bug source; do NOT drop it. Use `torch::max(x, -1)` returning a `(values, indices)` tuple.
5. **_combine_max / _median_smoothing** — `_combine_max`: `mask = a.first > thresh*b.first;`
   `values = where(mask, a.first, b.first); indices = where(mask, a.second, b.second);`.
   `_median_smoothing`: `pad_length = (win_length-1)/2;` left-pad by REPLICATING `indices[..., pad_length]`
   (`cat(pad_length copies, -1)` then concat the tail), `unfold(-1, win_length, 1)`,
   return `torch::median(roll, -1).values`.
6. **detect_pitch_frequency** — pack to `(-1, time)`; `nccf = _compute_nccf(...)`;
   `indices = _find_max_per_frame(nccf, sr, freq_high)`; `indices = _median_smoothing(indices, win_length)`;
   `freq = sample_rate / (EPSILON + indices.to(float));` reshape back to the input's leading dims.
7. **Tests + golden + coverage** — extend `gen_golden.py`:
   - `sliding_window_cmn` on a small random/ramp specgram for all four flag combos
     (center∈{F,T} × norm_vars∈{F,T}) plus a case exercising the `min_cmn_window` latency and the
     `window_end > num_frames` end clamp; emit shape/sum/sample values.
   - `detect_pitch_frequency` on synthetic 220 Hz and 440 Hz sinusoids at sr=16000 (assert the median
     estimate ≈ truth within a few Hz) and a direct check of `_compute_nccf`/`_median_smoothing`
     against torchaudio's internal helpers.
   Run `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`,
   bake the printed constants into `main.cpp`, register the new test functions in `main()`. Then:
   `cmake --build build --target audio_test_functional && ./build/unit_test/audio/functional/audio_test_functional`
   and `ctest --test-dir build` must be green; confirm 100% line coverage of the new lines per the
   coverage recipe (`-DTORCHMEDIA_COVERAGE=ON`, `llvm-cov`, `--ignore-filename-regex='_vendor/.*'`).

## Acceptance criteria
- [ ] `sliding_window_cmn` matches torchaudio within `TM_CHECK_TENSOR_CLOSE(atol=1e-4, rtol=1e-4)`
      for all four (center, norm_vars) combos AND on the `min_cmn_window` / end-clamp edge case.
- [ ] `detect_pitch_frequency` on 220 Hz and 440 Hz sinusoids returns a median estimate within a few
      Hz of truth; `_compute_nccf` and `_median_smoothing` match torchaudio internals within tolerance.
- [ ] The `_find_max_per_frame` `+1` offset is present and verified by the pitch golden test.
- [ ] Both option structs default to torchaudio's values; setters are exercised by a test.
- [ ] `ctest --test-dir build` green; 100% line coverage of the new lines in `_functional.hpp`.

## Constraints
- Header-only, `inline`; torch-native ATen ops only (`cumsum`, `unfold`, `torch::max`,
  `torch::median`, `torch::where`, `torch::linalg::vector_norm`, slice/narrow assign). No custom
  compiled op, no system deps.
- Both ops are inherently SEQUENTIAL — transcribe the per-frame (cmn) and per-lag (nccf) loops
  faithfully; the incremental `cur_sum`/`cur_sumsq` add/subtract and the window-index clamps are the
  primary correctness risk and must mirror torchaudio exactly.
- Match torchaudio's validation/behavior (e.g. 2D→squeeze leading dim, `EPSILON=1e-9`); raise via
  `handle_exceptions<...>` / `TORCH_CHECK` only where torchaudio raises.
- CI/coverage must not depend on `.venv` at build time — golden values are baked literals.

## Notes / Assumptions
- Assumption: no dependency on other tasks; both ops are torch-native and self-contained (independent
  per D6). They can be implemented in any order relative to task01/task02.
- Assumption: `linalg::vector_norm` with `ord=2` over the last dim is available in the pinned libtorch
  2.5.1; if the C++ overload is awkward, `(s*s).sum(-1).sqrt()` is an equivalent fallback (keep the
  `EPSILON` placement identical: `(EPSILON + norm).pow(2)`).
- Gotcha: torchaudio's `pow(variance, -0.5)` is undefined for single-frame windows (`W==1`,
  variance==0); zero those scales to avoid NaN/Inf and match torchaudio's effective output.
- Gotcha: the `+1` calibration offset in `_find_max_per_frame` and the centered-vs-left
  `window_start`/`window_end` clamp (incl. the `max(t+1, min_cmn_window)` non-center rule) are the
  documented bug sources — cover them explicitly with golden edge cases.
- Question for Mux: expose the four `_*` helpers as public `inline` functions for unit testing, or
  keep them in a `detail`/anonymous-namespace and test only through the public ops? Defaulting to
  public `inline` helpers (so `test_nccf_median_helpers` can call them directly); flag at review.
