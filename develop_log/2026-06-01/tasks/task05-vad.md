# Task 05 — Implement vad (voice activity detection, SoX cepstral-power state machine)
id: 2026-06-01/task05
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Add torch-native `vad` (SoX `vad` effect: leading-silence trim via an adaptive cepstral-power state
machine) plus its `_measure` helper, appended to `_functional_filtering.hpp`, matching torchaudio
v2.5.1 trim length exactly, with a `.venv` golden regression test and 100% coverage of the new lines.

## Scope
In:
- `vad(const torch::Tensor& waveform, int64_t sample_rate, const vad_option& opt = {})` — the public
  entry, returning the input trimmed of leading silence (or an empty `[..., 0]` tensor if it never
  triggers).
- `_measure(...)` — the per-window cepstral-power helper (effectively a second function; budget it as
  such). Keep it `inline` in the same header (file-local detail; it may live in a `detail`/anon
  namespace).
- `vad_option` struct in `_functional_methods_options.hpp` with fluent setters for all 17 SoX knobs.
- Golden test cross-checking the **returned trim length** vs torchaudio on silence-then-tone signals,
  plus a `_measure`-in-isolation unit test vs the `.venv` helper.

Out:
- Autograd / backward (forward-only; the trim is a pure index slice, no grad needed).
- Vectorization across windows (the outer loop is sequential by construction — do not attempt it).
- Trailing-silence trimming / reverse application (SoX/torchaudio leave that to running `vad` on a
  reversed signal at the call site; not part of this function).
- Any FFmpeg/sox runtime dependency.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D2 (header lives in
   `_functional_filtering.hpp`), D4 (`vad` is **scope-pending, recommended INCLUDE as its own task**),
   D5 (testing), and the Gotcha: "`vad` integer bin/index math (`dft_len_ws`, spectrum/cepstrum
   ranges) must match exactly or the trim index diverges."
2. `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/filtering.py` —
   authoritative source for `vad` and `_measure`. Port the integer/index math verbatim.
3. `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` — copy the
   `xxx_option` + fluent-setter pattern (e.g. `spectrogram_option`) for `vad_option`.

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp` (APPEND `vad` + `_measure`;
   created by task01; if task01 has not landed, create the header with the project include guard /
   namespace skeleton and append).
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` (ADD `vad_option`).
- `unit_test/audio/functional/main.cpp` (ADD tests + register in `main()`).
- `unit_test/audio/functional/gen_golden.py` (ADD a `vad` / `_measure` golden generator).
- `libtorchmedia/include/torchmedia.hpp` (only if `_functional_filtering.hpp` is not yet aggregated).

## Deliverables
- `vad` in `_functional_filtering.hpp`, signature:
  `inline torch::Tensor vad(const torch::Tensor& waveform, int64_t sample_rate, const vad_option& opt = {});`
  in `namespace torchmedia::audio::functional`.
- `_measure` in the same header (inline; returns `double`/`float` — one scalar per channel-window).
- `vad_option` in `_functional_methods_options.hpp` with fields + fluent setters (defaults from the
  torchaudio signature):
  `trigger_level=7.0, trigger_time=0.25, search_time=1.0, allowed_gap=0.25, pre_trigger_time=0.0,
  boot_time=0.35, noise_up_time=0.1, noise_down_time=0.01, noise_reduction_amount=1.35,
  measure_freq=20.0, measure_duration=std::optional<double>{} (None), measure_smooth_time=0.4,
  hp_filter_freq=50.0, lp_filter_freq=6000.0, hp_lifter_freq=150.0, lp_lifter_freq=2000.0`.
- Assertion tests in `main.cpp` (length-exact + `_measure` point-wise) and the baked golden constants.

## Steps
1. **Add `vad_option`** — in `_functional_methods_options.hpp`, mirror the existing option-struct
   style: public fields above + a fluent setter per field (`vad_option& trigger_level(double);` …).
   `measure_duration` is `std::optional<double>` (None ⇒ `2.0 / measure_freq`).

2. **Port `_measure` first** (it is the inner kernel; everything else is bookkeeping around it).
   Signature mirrors torchaudio's helper; inputs are per-channel `spectrum`/`noise_spectrum`
   1-D buffers it mutates **in place**, the precomputed windows/bin ranges, and `boot_count`.
   Body (ATen-for-ATen with the Python):
   - `dft_len_ws = spectrum.size(-1)`; `TORCH_CHECK(spectrum.size(-1) == noise_spectrum.size(-1), ...)`.
   - `dftBuf = zeros(dft_len_ws)`; `dftBuf[:measure_len_ws] = samples * spectrum_window[:measure_len_ws]`.
   - `_dftBuf = torch::fft::rfft(dftBuf)`.
   - `mult = boot_count >= 0 ? boot_count / (1.0 + boot_count) : measure_smooth_time_mult`.
   - `_d = _dftBuf[spectrum_start:spectrum_end].abs()`;
     `spectrum[start:end].mul_(mult).add_(_d * (1 - mult))`; `_d = spectrum[start:end].pow(2)`.
   - `_mult = boot_count >= 0 ? zeros : torch::where(_d > noise_spectrum[start:end], noise_up_time_mult, noise_down_time_mult)`.
   - `noise_spectrum[start:end].mul_(_mult).add_(_d * (1 - _mult))`.
   - `_d = sqrt(max(zeros, _d - noise_reduction_amount * noise_spectrum[start:end]))`.
   - `_cepstrum_Buf = zeros(dft_len_ws >> 1)`; `[start:end] = _d * cepstrum_window`; tail zeroed.
   - `_cepstrum_Buf = torch::fft::rfft(_cepstrum_Buf)`.
   - `result = sum(_cepstrum_Buf[cepstrum_start:cepstrum_end].abs().pow(2)).item<double>()`.
   - `result = result > 0 ? std::log(result / (cepstrum_end - cepstrum_start)) : -inf`;
     return `std::max(0.0, 21 + result)`. **Note the two rffts.**

3. **Precompute the constants in `vad`** (integer math must match exactly — these set the trim index):
   - `measure_duration = opt.measure_duration ? *opt.measure_duration : 2.0 / measure_freq`.
   - `measure_len_ws = int(sr * measure_duration + 0.5)`; `measure_len_ns = measure_len_ws`.
   - `dft_len_ws = 16; while (dft_len_ws < measure_len_ws) dft_len_ws *= 2;` (next pow2 ≥ 16).
   - `measure_period_ns = int(sr / measure_freq + 0.5)`.
   - `measures_len = ceil(search_time * measure_freq)`; `search_pre_trigger_len_ns = measures_len * measure_period_ns`.
   - `gap_len = int(allowed_gap * measure_freq + 0.5)`.
   - `fixed_pre_trigger_len_ns = int(pre_trigger_time * sr + 0.5)`.
   - `samplesLen_ns = fixed_pre_trigger_len_ns + search_pre_trigger_len_ns + measure_len_ns`.
   - `spectrum_window = full(measure_len_ws, 2/sqrt(measure_len_ws)) * hann_window(measure_len_ws)`.
   - `spectrum_start = max(int(hp_filter_freq/sr*dft_len_ws + 0.5), 1)`;
     `spectrum_end = min(int(lp_filter_freq/sr*dft_len_ws + 0.5), dft_len_ws/2)`.
   - `cepstrum_window = full(spectrum_end-spectrum_start, 2/sqrt(spectrum_end-spectrum_start)) * hann_window(spectrum_end-spectrum_start)`.
   - `cepstrum_start = ceil(sr*0.5/lp_lifter_freq)`; `cepstrum_end = min(floor(sr*0.5/hp_lifter_freq), dft_len_ws/4)`.
   - exp multipliers: `noise_up_time_mult = exp(-1/(noise_up_time*measure_freq))`,
     `noise_down_time_mult = exp(-1/(noise_down_time*measure_freq))` (as scalar tensors for `where`),
     `measure_smooth_time_mult = exp(-1/(measure_smooth_time*measure_freq))`,
     `trigger_meas_time_mult = exp(-1/(trigger_time*measure_freq))`.
   - `boot_count_max = int(boot_time*measure_freq - 0.5)`; `boot_count = measures_index = flushedLen_ns = 0`.

4. **Shape handling** — `auto shape = waveform.sizes()`; `waveform.view({-1, shape.back()})`;
   `n_channels, ilen`. ndim>2 ⇒ **only warn** (do not raise) — torchaudio flattens leading dims.
   Allocate per-channel state: `mean_meas`, `spectrum`, `noise_spectrum` (`[n_channels, dft_len_ws]`),
   `measures` (`[n_channels, measures_len]`), all zeros.

5. **Outer per-window loop** — `for (pos = measure_len_ns; pos < ilen; pos += measure_period_ns)`:
   - For each channel `i`: `meas = _measure(... samples = waveform[i, pos-measure_len_ws : pos] ...)`;
     `measures[i, measures_index] = meas`;
     `mean_meas[i] = mean_meas[i]*trigger_meas_time_mult + meas*(1 - trigger_meas_time_mult)`.
   - `has_triggered = has_triggered || (mean_meas[i].item<double>() >= trigger_level)`.
   - **Inner backward search** (only when `has_triggered`): with `n=measures_len`, `k=measures_index`,
     `jTrigger=jZero=n`, walk `for (j=0; j<n; ++j)`:
       - if `measures[i,k] >= trigger_level && j <= jTrigger + gap_len`: `jZero = jTrigger = j`.
       - elif `measures[i,k] == 0 && jTrigger >= jZero`: `jZero = j`.
       - `k = (k + n - 1) % n`.
     After the loop: `j = min(j, jZero)`; `num_measures_to_flush = min(max(num_measures_to_flush, j), n)`.
     (`j` after a C++ `for` ends at `n`; match Python's post-loop `j` value — i.e. use `n`.)
   - After the channel loop: `measures_index = (measures_index + 1) % measures_len`;
     `if (boot_count >= 0) boot_count = (boot_count == boot_count_max) ? -1 : boot_count + 1`.
   - `if (has_triggered) { flushedLen_ns = (measures_len - num_measures_to_flush) * measure_period_ns; break; }`.

6. **Return / trim** — if never triggered: return `waveform[..., 0:0]` reshaped to `shape[:-1] + {0}`.
   Else `res = waveform[:, pos - samplesLen_ns + flushedLen_ns :]` reshaped to `shape[:-1] + {res.size(-1)}`.
   Use `.index({Slice(), Slice(start)})` / `.view(...)` to restore the original leading dims.

7. **Errors / validation** — `TORCH_CHECK` floating dtype; the `_measure` spectrum-size check; route
   public failures through the project `handle_exceptions<...>` helper if task01 established that
   pattern in this header (match its style). Match torchaudio: ndim>2 warns, does not raise.

8. **Tests, golden, ctest, coverage** — extend `gen_golden.py` to (a) run `torchaudio.functional.vad`
   on 2-3 deterministic signals (e.g. `sr=16000`: leading silence of N samples then a 440 Hz tone;
   one all-silence case expecting an empty return) and print the **exact returned length** (and the
   leading sample offset) as C++ constants; (b) call the private `torchaudio.functional.filtering._measure`
   (or replicate its inputs) for a couple of windows and print the scalar results. Run
   `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`,
   bake constants into `main.cpp`. Add `test_vad()` + `test_measure()` using `TM_CHECK` /
   `TM_CHECK_CLOSE`; assert returned length is **exactly equal** (integer), and `_measure` scalars
   match within `1e-5`. Register both in `main()`'s call list. Build & run:
   `cmake --build build --target audio_test_functional && ./build/unit_test/audio/functional/audio_test_functional`;
   `ctest --test-dir build` green; confirm 100% line coverage of the new `vad`/`_measure` lines
   (`-DTORCHMEDIA_COVERAGE=ON`, `llvm-cov` with `--ignore-filename-regex='_vendor/.*'`).

## Acceptance criteria
- [ ] `vad` returns a tensor whose **trim length and leading offset exactly match** torchaudio v2.5.1
      on the golden silence-then-tone cases (integer-exact, not approximate).
- [ ] All-silence (never-triggered) input returns an empty `[..., 0]` tensor with original leading dims.
- [ ] `_measure` scalar output matches the `.venv` helper within `1e-5` on the golden windows.
- [ ] `vad_option` exposes all 17 knobs with fluent setters and torchaudio defaults; `measure_duration`
      None-default resolves to `2/measure_freq`.
- [ ] Multi-channel (`[C, N]`) input handled; ndim>2 warns (does not raise) and matches torchaudio.
- [ ] `ctest --test-dir build` green; 100% line coverage of the new `vad`/`_measure` lines.

## Constraints
- **Header-only**, `inline` free functions, `namespace torchmedia::audio::functional`; torch-native
  ATen only (`torch::fft::rfft`, `torch::hann_window`, `torch::where`, `torch::max`, in-place
  `mul_`/`add_`, `pow`, `sqrt`, indexing `Slice`) — no custom op, no backend, no FFmpeg/sox.
- **Sequential by construction**: the outer window loop carries `spectrum`/`noise_spectrum`/`mean_meas`
  state and `boot_count`; the inner search reads the circular `measures` ring backward — do not
  vectorize or reorder.
- **Integer/index math must match torchaudio bit-for-bit** (`int(... + 0.5)` truncation, `ceil`/`floor`,
  `dft_len_ws` next-pow2 via while-shift, `dft_len_ws/2`/`/4` floor div). Any divergence shifts the
  final trim index and fails the length-exact assertion.
- Golden constants are baked into `main.cpp`; **the build must not depend on `.venv`** at compile/test
  time (D1/Gotcha in progress01).
- **Scope-pending (D4):** this task is recommended-INCLUDE but Mux confirms inclusion at review.

## Notes / Assumptions
- Assumption: no dependency on other tasks for the algorithm itself (`_measure` is self-contained);
  the only coupling is that `vad` lands in `_functional_filtering.hpp`, created by **task01** (lfilter).
  If task01 has not landed yet, create the header with the standard guard/namespace skeleton so this
  task is not blocked.
- Assumption: per-channel scalar reads (`mean_meas[i].item<double>()`, `measures[i,k]`) are acceptable
  inside the loop; the inner kernel cost is dominated by the two rffts, so `.item()` overhead is moot.
- Gotcha: `_measure` does **two** `rfft`s (spectrum, then cepstrum); the cepstrum buffer is sized
  `dft_len_ws >> 1`, and only `[spectrum_start:spectrum_end]` is filled before the tail is zeroed.
- Gotcha: the post-loop `j` in Python equals `n` (range exhausted) — `num_measures_to_flush` uses
  `min(j, jZero)` with `j==n`; in C++ use the loop bound `n`, not the last `j` value, to match.
- Gotcha: `boot_count` flips to `-1` exactly once it reaches `boot_count_max`, switching `_measure`'s
  `mult` from the boot ramp `boot_count/(1+boot_count)` to the steady `measure_smooth_time_mult` and
  enabling the adaptive noise `where`.
- Question for Mux: confirm `vad` is IN for this milestone (D4 scope-pending). If deferred, mark this
  task `blocked` and skip implementation. Also confirm the public surface is `vad(waveform, sr, opt)`
  with `vad_option` (vs. a flat positional signature mirroring torchaudio) — default assumption is the
  project's `xxx_option` convention.
