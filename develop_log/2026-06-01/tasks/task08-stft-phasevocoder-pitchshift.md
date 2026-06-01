# Task 08 — STFT domain: inverse_spectrogram, phase_vocoder, pitch_shift
id: 2026-06-01/task08
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Add torch-native `inverse_spectrogram`, `phase_vocoder`, and `pitch_shift` (the latter built on the
former two) to `_functional.hpp`, each with a torchaudio 2.5.1 golden test and 100% coverage.

## Scope
In:
- `inverse_spectrogram` + `inverse_spectrogram_option` (the ISTFT counterpart to the existing
  `spectrogram`).
- `phase_vocoder(complex_specgrams, rate, phase_advance)` — fully vectorized, no time loop.
- `pitch_shift` + `pitch_shift_option` — `_stretch_waveform` (stft → `phase_vocoder` → istft) then
  `resample` then `_fix_waveform_shape` crop/pad.
- Golden assertion tests for all three; new golden constants in `gen_golden.py`.
Out:
- Autograd / backward (forward-only; rely on ATen autodiff if a tensor carries grad, no custom op).
- Non-Hann windows as a *default* (a caller-supplied window is honored; the implicit default stays
  `hann_window`, matching torchaudio).
- The `transform` wrappers (`InverseSpectrogram`, `PitchShift`, `TimeStretch`) — later transform progress.
- Resampling method other than the existing `resample` (Hann sinc only — see task05/resample).

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D1 (.venv golden), D2 (file org:
   feature ops extend `_functional.hpp`), D5 (testing), D6 (task08 must port `phase_vocoder` before
   `pitch_shift`, same task).
2. torchaudio v2.5.1 source (authoritative algorithm):
   `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/functional.py`
   — read `inverse_spectrogram`, `phase_vocoder`, `pitch_shift`, `_stretch_waveform`,
   `_fix_waveform_shape`, and `_get_spec_norms`. Load `WebFetch` via `ToolSearch "select:WebFetch"`
   if you need the exact lines.

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — existing `spectrogram` (lines ~149-181,
   shows window default + `_normalize_method` "window"/"frame_length" handling), `griffinlim`
   (istft usage), `resample` (signature `resample(wav, orig_freq, new_freq, opt)`), and
   `handle_exceptions<T, ExceptionT>(...)`. Append the three new functions here.
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` — add
   `inverse_spectrogram_option` and `pitch_shift_option` next to `spectrogram_option` (lines 10-83),
   using the same fluent `xxx_option &` setter style.
- `unit_test/audio/functional/main.cpp` — add tests + register in `main()` (the call list ends at
   `test_option_setters()` / line ~340).
- `unit_test/audio/functional/gen_golden.py` — append a phase_vocoder + pitch_shift golden block.

## Deliverables
- In `_functional.hpp` (all `inline`, namespace `torchmedia::audio::functional`):
  - `inline auto inverse_spectrogram(tensor_t spectrogram, std::optional<int64_t> length, const inverse_spectrogram_option &opt) -> tensor_t;`
  - `inline auto phase_vocoder(tensor_t complex_specgrams, double rate, const_tensor_lref_t phase_advance) -> tensor_t;`
  - `inline auto pitch_shift(const_tensor_lref_t waveform, int sample_rate, double n_steps, const pitch_shift_option &opt = {}) -> tensor_t;`
  - Internal helpers `_stretch_waveform(...)` and `_fix_waveform_shape(...)` (inline, file-local
    naming, mirroring torchaudio's privates).
- In `_functional_methods_options.hpp`:
  - `inverse_spectrogram_option` (fields: `_pad`, `_window`, `_n_fft=400`, `_hop_length=200`,
    `_win_length=400`, `_normalized=false`, `_normalize_method="window"`, `_center=true`,
    `_pad_mode="reflect"`, `_onesided=true`) with fluent setters.
  - `pitch_shift_option` (fields: `_bins_per_octave=12`, `_n_fft=512`, `_win_length` (optional/0=default),
    `_hop_length` (optional/0=default), `_window` (optional/undefined=Hann)) with fluent setters.
- In `main.cpp`: `test_inverse_spectrogram_roundtrip`, `test_inverse_spectrogram_branches`,
  `test_phase_vocoder`, `test_pitch_shift`, registered in `main()`.
- In `gen_golden.py`: a block printing the phase_vocoder + pitch_shift constants baked into `main.cpp`.

## Steps
1. **`_get_spec_norms` resolution** — map `_normalized` + `_normalize_method` to the two bools
   torchaudio uses: `frame_length_norm` (true when `normalize_method=="frame_length"`) and
   `window_norm` (true when `normalize_method=="window"`); only one is active and only if
   `_normalized`. This is the inverse of the spectrogram normalization at lines 170-176.
2. **`inverse_spectrogram`** — `TORCH_CHECK(spectrogram.is_complex(), ...)` (raise like torchaudio).
   Resolve `window` (default `torch::hann_window(win_length, dtype/device of a real view of the input)`).
   If `window_norm`, pre-multiply `spectrogram = spectrogram * window.pow(2).sum().sqrt()` (undo the
   forward `/ sqrt(sum win^2)`). Pack leading dims: keep last two as `(freq, time)`, reshape to
   `(-1, freq, time)`. Call `torch::istft(packed, n_fft, hop_length, win_length, window, center,
   /*normalized=*/frame_length_norm, onesided, /*length=*/<see below>, /*return_complex=*/false)`.
   Length handling: if `length` given, pass `length + 2*pad` to istft; then `slice(-1, pad, pad+length)`
   to strip `pad` samples each end; if no `length`, pass `std::nullopt` and skip the strip. Reshape
   back to the original leading dims + the recovered time axis. Note: `pad_mode` is accepted for API
   parity but unused (document the no-op).
3. **`phase_vocoder`** — early return `complex_specgrams` if `rate == 1.0`.
   Pack to `(-1, freq, n_frames)`. `time_steps = torch::arange(0, n_frames, rate, ...)` (float).
   `alphas = time_steps % 1` (use `torch::fmod` / `time_steps - time_steps.floor()`);
   `phase_0 = torch::angle(specgrams.slice(time, 0, 1))` (angle of the first frame).
   Pad the time axis by 2 on the right (`constant_pad_nd({0,2})`). Gather two neighbor frames:
   `idx0 = time_steps.floor().to(kLong)`, `spec0 = index_select(time, idx0)`,
   `spec1 = index_select(time, idx0 + 1)`. `phase = torch::angle(spec1) - torch::angle(spec0)
   - phase_advance`; wrap to (-pi, pi]: `phase = phase - 2*pi*torch::round(phase / (2*pi))`;
   add back `phase_advance`. Prepend `phase_0` along time, `phase_acc = torch::cumsum(phase, time)`.
   Magnitude: `mag = alphas * torch::abs(spec1) + (1 - alphas) * torch::abs(spec0)`.
   `out = torch::polar(mag, phase_acc)`. Unpack to original leading dims with new `n_frames ==
   time_steps.numel()`. (Vectorized — assert no time loop is used.)
4. **`pitch_shift` / `_stretch_waveform`** — defaults: `hop_length = win_length/4` and
   `win_length = n_fft` when unset; window defaults to `hann_window(win_length)`. Compute
   `ori_len = waveform.size(-1)`; `rate = std::pow(2.0, -n_steps / bins_per_octave)`.
   `spec = torch::stft(waveform, n_fft, hop, win, window, /*center=*/true, "reflect",
   /*normalized=*/false, /*onesided=*/true, /*return_complex=*/true)`.
   `phase_advance = torch::linspace(0, pi*hop, n_freq).unsqueeze(-1)` where `n_freq = n_fft/2 + 1`.
   `spec_stretch = phase_vocoder(spec, rate, phase_advance)`;
   `len_stretch = (int64_t)std::round(ori_len / rate)`;
   `waveform_stretch = torch::istft(spec_stretch, n_fft, hop, win, window, true, false, true,
   /*length=*/len_stretch)`.
5. **`pitch_shift` resample + fix shape** — `waveform_shift = resample(waveform_stretch,
   (int)std::round(sample_rate / rate), sample_rate)`; then `_fix_waveform_shape`: if
   `waveform_shift.size(-1) > ori_len` crop `slice(-1, 0, ori_len)`, else right-pad with zeros to
   `ori_len` (`constant_pad_nd({0, ori_len - shifted_len})`). Return.
6. **Options + parity** — add the two option structs with fluent setters; validate inputs to match
   torchaudio raises (complex-input check in `inverse_spectrogram`; accept but no-op `pad_mode`).
7. **Tests, golden, coverage** —
   - `test_inverse_spectrogram_roundtrip`: build a real signal, `spectrogram(..., return_complex=true)`
     with COLA-satisfying window (`hann`, `hop=n_fft/4`, `center=true`), then
     `inverse_spectrogram(spec, length=N, ...)` and `TM_CHECK_TENSOR_CLOSE(recon, original, atol, rtol)`
     within ISTFT COLA tolerance; also test the no-`length` path shape.
   - `test_inverse_spectrogram_branches`: `window`/`frame_length`/no-norm paths; `pad>0` strip; assert
     the non-complex input raises (catch like the existing error tests, e.g. `test_convolve_*`).
   - `test_phase_vocoder`: on a fixed random complex spec with fixed `phase_advance`, assert (a) identity
     at `rate==1.0` (`TM_CHECK(torch::equal(...))`), (b) output frame count `== arange(0,T,rate).numel()`
     for `rate` up (e.g. 1.3) and down (0.7), (c) element values match `.venv` golden constants.
   - `test_pitch_shift`: end-to-end vs `.venv` `torchaudio.functional.pitch_shift` golden, output length
     `== ori_len`, FP tolerance.
   - Add golden generation to `gen_golden.py` (random spec with a fixed seed; print a few output
     elements + frame counts; pitch_shift on a fixed sine). Run
     `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`
     and bake the printed constants into `main.cpp` (no runtime .venv dependency).
   - Register all four tests in `main()`. Build & run:
     `cmake --build build --target audio_test_functional && ./build/unit_test/audio/functional/audio_test_functional`;
     `ctest --test-dir build` must be green; confirm 100% line coverage of the newly added lines in
     `_functional.hpp` (vendored `_vendor/` excluded).

## Acceptance criteria
- [ ] `inverse_spectrogram` round-trips a COLA-satisfying complex spectrogram back to the source
      waveform within ISTFT tolerance; `length`/no-`length` and `pad>0` strip paths exercised; complex
      input check raises on a real tensor.
- [ ] `phase_vocoder` returns input unchanged at `rate==1.0`; produces `arange(0,T,rate).numel()`
      frames for rate up & down; matches `.venv` golden elementwise within tolerance; no time loop.
- [ ] `pitch_shift` output length equals the original; values match `.venv`
      `torchaudio.functional.pitch_shift` within FP tolerance for an up- and a down-shift.
- [ ] Both option structs present with fluent setters; covered by a setter test (extend
      `test_option_setters` or add one).
- [ ] `ctest --test-dir build` green; 100% line coverage of the new `_functional.hpp` lines.

## Constraints
- Header-only, `inline`, namespace `torchmedia::audio::functional`; torch-native ATen ops only
  (`torch::istft`, `torch::stft`, `torch::angle`, `torch::abs`, `torch::round`, `torch::cumsum`,
  `torch::polar`, `index_select`, `torch::fmod`/`floor`, `torch::arange`, `torch::linspace`,
  `constant_pad_nd`). No new third-party deps.
- Match torchaudio v2.5.1 validation/raises: `inverse_spectrogram` requires a complex tensor; the
  `normalized` str↔bool resolution must mirror `_get_spec_norms` and stay the exact inverse of the
  forward `spectrogram` normalization (lines 170-176).
- `phase_vocoder` must be fully vectorized (no per-frame C++ loop) — this is its whole point.
- `n_steps` is a `double` (fractional semitone shifts are valid); `rate = 2^(-n_steps/bins_per_octave)`.
- `pad_mode` is accepted on `inverse_spectrogram_option` for parity but is a documented no-op (istft
  has no pad_mode).
- Golden constants baked into `main.cpp`; the build/coverage runs must not invoke `.venv`.

## Notes / Assumptions
- Assumption: `pitch_shift` may reuse the existing `resample` (Hann sinc, task05). Small numeric
  differences vs torchaudio's default resample are tolerable — pick `atol/rtol` from the observed
  `.venv` delta; if it drifts too far, prefer matching torchaudio's `resample` call signature
  (`int(sample_rate/rate) → sample_rate`) exactly and widen tolerance rather than changing the kernel.
- Assumption: `inverse_spectrogram` reuses the same window default and normalization semantics as the
  existing `spectrogram` (so a `spectrogram(return_complex=true)` → `inverse_spectrogram` round-trip is
  consistent end-to-end).
- Assumption: an "onesided" complex input is the common case; `torch::istft` requires onesided unless
  `onesided=false`, so pass the option through unchanged.
- Question for Mux: keep the `pad_mode` field on `inverse_spectrogram_option` (API parity, no-op) or
  drop it to avoid implying behavior it doesn't have? Default: keep, documented as no-op.
- Dependency: this task is self-contained (does not need task01 lfilter). It depends only on the
  already-shipped `spectrogram`, `griffinlim`/istft usage pattern, and `resample`; per D6, `phase_vocoder`
  is implemented before `pitch_shift` within this task.
