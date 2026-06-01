# Task 09 — Implement compute_deltas, linear_fbanks, spectral_centroid
id: 2026-06-01/task09
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Add three torch-native feature ops — `compute_deltas`, `linear_fbanks`, and `spectral_centroid` —
to `_functional.hpp`, mirroring `torchaudio.functional` v2.5.1, each with golden regression tests
and 100% line coverage.

## Scope
In:
- `compute_deltas(specgram, win_length=5, mode="replicate")` — depthwise conv1d delta filter.
- `linear_fbanks(n_freqs, f_min, f_max, n_filter, sample_rate)` — linear-Hz triangular filterbank.
- `spectral_centroid(waveform, sample_rate, pad, window, n_fft, hop_length, win_length)` — magnitude-
  weighted mean frequency per frame (depends on existing `spectrogram`).
- A shared `_create_triangular_filterbank(all_freqs, f_pts)` helper, factored out so both this op and
  the existing `mel_filter_bank` could use it (see Notes — mel reuse is optional this task).

Out:
- Autograd / backward (forward-only, no custom `torch::autograd::Function`).
- Non-`replicate` `compute_deltas` pad modes beyond what `torch::nn::functional::pad` supports natively
  (we still accept any `mode` string and pass it through, but only `replicate` is golden-tested).
- Rewriting `mel_filter_bank`'s existing nested-loop construction (leave it as-is; only *add* the
  shared helper — refactoring mel to use it is a nice-to-have, not required).
- Any `transform`-layer wrappers (`ComputeDeltas`, `LinearFbank`, `SpectralCentroid`) — later progress.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D2 (extend `_functional.hpp`,
   options in `_functional_methods_options.hpp`), D5 (testing/coverage), D6 (task09 is independent).
2. torchaudio v2.5.1 source (signatures, algorithm, raises, defaults):
   `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/functional.py`
   (functions `compute_deltas`, `linear_fbanks`, `spectral_centroid`, `_create_triangular_filterbank`).

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — append the three ops + the helper;
  reuse the existing `spectrogram` for `spectral_centroid`; see `mel_filter_bank` (lines ~184-270)
  for the triangular-filter shape this mirrors and the `linspace(0, sample_rate/2, ...)` bin grid.
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` — add option struct(s)
  only if you choose the option-struct form (see Steps 1/4; plain args are acceptable here).
- `unit_test/audio/functional/main.cpp` — add tests + register in `main()`.
- `unit_test/audio/functional/gen_golden.py` — append golden generators for the cross-check constants.

## Deliverables
- In `libtorchmedia/include/torchmedia/_audio/_functional.hpp`, inline free functions in namespace
  `torchmedia::audio::functional`:
  - `inline auto _create_triangular_filterbank(const_tensor_lref_t all_freqs, const_tensor_lref_t f_pts) -> tensor_t;`
    returns `(n_freqs, n_filter)`.
  - `inline auto compute_deltas(const_tensor_lref_t specgram, int win_length = 5, const std::string &mode = "replicate") -> tensor_t;`
    returns same shape `(..., freq, time)`.
  - `inline auto linear_fbanks(int n_freqs, double f_min, double f_max, int n_filter, int sample_rate) -> tensor_t;`
    returns `(n_freqs, n_filter)`.
  - `inline auto spectral_centroid(const_tensor_lref_t waveform, int sample_rate, int pad, const_tensor_lref_t window, int n_fft, int hop_length, int win_length) -> tensor_t;`
    returns `(..., time)`. (Plain args mirror torchaudio's positional signature; an optional
    `spectral_centroid_option` is acceptable but not required — pick one and keep it consistent.)
- Assertion tests in `unit_test/audio/functional/main.cpp` (registered in `main()`), plus appended
  golden generators in `gen_golden.py` with the printed constants baked into `main.cpp` as literals.

## Steps
1. **Triangular helper** — port `_create_triangular_filterbank(all_freqs, f_pts)` exactly:
   `f_diff = f_pts[1:] - f_pts[:-1]`; `slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)`;
   `down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]`; `up_slopes = slopes[:, 2:] / f_diff[1:]`;
   `fb = max(0, min(down_slopes, up_slopes))`. Use `torch::min`, `torch::max`/`clamp_min(0)`,
   `.slice(1, ...)`, `.unsqueeze(0/1)`. Result shape `(n_freqs, n_filter)`.
2. **linear_fbanks** — `all_freqs = torch::linspace(0, sample_rate/2 (integer floor), n_freqs)`;
   `f_pts = torch::linspace(f_min, f_max, n_filter + 2)` (LINEAR Hz mid-points, no mel mapping);
   return `_create_triangular_filterbank(all_freqs, f_pts)`. Match torchaudio's `sample_rate // 2`
   (integer floor div) for the upper `linspace` endpoint.
3. **compute_deltas** — `shape = specgram.sizes()`; reshape to `(1, -1, shape[-1])`;
   `TORCH_CHECK(win_length >= 3, ...)` (mirror torchaudio's ValueError message);
   `n = (win_length - 1) / 2`; `denom = n*(n+1)*(2*n+1)/3.0`;
   pad time by `n` each side via `torch::nn::functional::pad(..., PadFuncOptions({n, n}).mode(...))`
   with the requested `mode` (`replicate` default, torch enum `torch::kReplicate`);
   build `kernel = torch::arange(-n, n+1, dtype=specgram.dtype)` reshaped to `(1,1,win_length)` and
   `.repeat({channels, 1, 1})` (channels = `specgram.size(1)` after reshape) to get `(channels,1,win_length)`;
   `output = conv1d(specgram, kernel, Conv1dFuncOptions().groups(channels)) / denom`;
   reshape back to original `shape`. Note: kernel is `arange(-n..n)`, depthwise, divided by `denom`
   (no extra normalization). Use `torch::nn::functional::conv1d`.
4. **spectral_centroid** — build a `spectrogram_option` with `.pad(pad).window(window).n_fft(n_fft)
   .hop_length(hop_length).win_length(win_length).power(1.0).normalized(false).return_complex(false)`
   (power=1.0 -> MAGNITUDE spectrogram) and call the existing `spectrogram(waveform, opt)`;
   `freqs = torch::linspace(0, sample_rate/2 (integer floor), 1 + n_fft/2).reshape({-1, 1})` (column,
   on `specgram.device()`); return `(freqs * specgram).sum(-2) / specgram.sum(-2)` (freq dim = -2).
5. **Tests + golden + green** — add `test_compute_deltas`, `test_linear_fbanks`,
   `test_spectral_centroid` to `main.cpp` and register them in `main()`:
   - `compute_deltas`: closed-form on a per-time-step linear ramp — for `specgram[...,t] = a*t + b`
     the interior delta equals the slope `a` exactly (boundary cols differ due to `replicate` pad);
     assert interior columns ~= slope; also assert output shape == input shape; cover the
     `win_length < 3` raise (expect `c10::Error`/exception) and a non-default `win_length` (e.g. 7).
   - `linear_fbanks`: small `n_filter` (e.g. `n_freqs=5, n_filter=2`) — assert shape `(5,2)`, all
     entries in `[0,1]`, each filter's peak == 1 at its center bin, and a torchaudio golden `.sum()`.
   - `spectral_centroid`: single-tone sine near a bin frequency (e.g. 440 Hz @ 16 kHz, n_fft=512) —
     assert returned shape `(1, time)` and that the centroid per frame is within a few bins (~tone
     freq); add a torchaudio golden mean/`[0,0]` constant.
   - Append generators to `gen_golden.py`, run
     `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`,
     bake the printed constants as literals (do NOT make the build depend on `.venv`).
   - Build & run: `cmake --build build --target audio_test_functional &&
     ./build/unit_test/audio/functional/audio_test_functional`; then `ctest --test-dir build` green.
   - Confirm 100% line coverage of the new lines (`-DTORCHMEDIA_COVERAGE=ON`, `llvm-profdata` +
     `llvm-cov` with `--ignore-filename-regex='_vendor/.*'`); cover both `compute_deltas` branches
     (the raise and the success path) and the helper's slope arithmetic.

## Acceptance criteria
- [ ] `compute_deltas` returns input shape; interior columns equal the ramp slope within `1e-4`;
      `win_length < 3` raises; matches torchaudio on a random spectrogram within `atol=1e-4, rtol=1e-4`.
- [ ] `linear_fbanks(n_freqs, f_min, f_max, n_filter, sample_rate)` shape == `(n_freqs, n_filter)`,
      entries in `[0,1]`, per-filter peak == 1, and `.sum()` matches torchaudio within tolerance.
- [ ] `spectral_centroid` shape == `(..., time)`; single-tone centroid within a few bins of the tone;
      matches torchaudio golden mean/`[0,0]` within tolerance.
- [ ] `ctest --test-dir build` green; new test functions registered in `main()`.
- [ ] 100% line coverage of the new lines in `_functional.hpp` (vendored excluded).

## Constraints
- Header-only: `inline` free functions in `_functional.hpp`, namespace `torchmedia::audio::functional`;
  run clang-format (LLVM, IndentWidth 4, ColumnLimit 120) before done.
- Torch-native ATen ops only: `torch::linspace`, `torch::arange`, `torch::min`/`torch::max`/`clamp_min`,
  `torch::nn::functional::pad`, `torch::nn::functional::conv1d`, tensor `.slice`/`.unsqueeze`/`.reshape`/
  `.repeat`/`.sum`. No system libraries.
- Match torchaudio's validation/raises: `compute_deltas` rejects `win_length < 3` (use `TORCH_CHECK`
  or the project `handle_exceptions<...>` helper, mirroring the upstream message).
- Integer floor for the upper `linspace` endpoint (`sample_rate / 2` truncated) in both `linear_fbanks`
  and `spectral_centroid`, matching torchaudio's `sample_rate // 2`.
- No sequential time loop is needed here (all three are vectorized); no complex linalg involved.
- Golden constants baked into `main.cpp`; the build must not depend on `.venv` at runtime.

## Notes / Assumptions
- Assumption: `spectrogram` (already in `_functional.hpp`) is correct and available; `spectral_centroid`
  reuses it with `power=1.0` (magnitude), so this task has NO dependency on task01 (lfilter) or any
  other task — it is independent (progress01 D6).
- Assumption: `tensor_t`, `const_tensor_lref_t`, and `tensor_options_t` aliases are available from
  `../globel_include.hpp` (already included by `_functional.hpp`).
- Assumption: depthwise conv1d with an `arange(-n..n)` kernel divided by `denom` IS torchaudio's exact
  delta (no separate `2*sum(n^2)` factor — `denom = n(n+1)(2n+1)/3` already equals `2*sum_{1..n} n^2`).
- Gotcha: `compute_deltas` kernel and the spectrogram must share dtype/device — build the kernel with
  `specgram.dtype()`/`specgram.device()`; cross-dtype conv1d will throw.
- Gotcha: `_create_triangular_filterbank` slicing — `slopes[:, :-2]` and `slopes[:, 2:]` use
  `.slice(1, 0, n_filter)` / `.slice(1, 2, n_filter+2)` (f_pts has `n_filter+2` points), and
  `f_diff[:-1]`/`f_diff[1:]` are `.slice(0, 0, n_filter)` / `.slice(0, 1, n_filter+1)`.
- Question for Mux: keep `mel_filter_bank` as-is, or refactor it to call the new shared
  `_create_triangular_filterbank` (would change mel internals — out of scope unless approved)? Default:
  add the helper, leave mel untouched.
