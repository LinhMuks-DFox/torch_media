# Task 07 — Convolution & augmentation: fftconvolve, add_noise, speed
id: 2026-06-01/task07
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Append three native-torch ops — `fftconvolve`, `add_noise`, and `speed` — to
`libtorchmedia/include/torchmedia/_audio/_functional.hpp`, mirroring torchaudio v2.5.1, each with
assertion tests and golden cross-checks.

## Scope
In:
- `fftconvolve(x, y, mode = full)` — FFT-domain linear convolution; reuses the existing
  `_apply_convolve_mode` crop helper (full/valid/same).
- `add_noise(waveform, noise, snr, lengths = {})` — scale noise to hit a requested per-signal SNR,
  with optional length masking for the energy estimate.
- `speed(waveform, orig_freq, factor, lengths = {}) -> std::tuple<torch::Tensor, c10::optional<torch::Tensor>>`
  — speed change as a `resample`, plus rescaled `out_lengths`.

Out:
- Autograd / backward (forward-only, no custom `torch::autograd::Function`).
- Any non-Hann resample method (`speed` delegates to the already-ported `resample`, Hann only).
- Compressed-codec or sox-effect speed paths; `phase_vocoder`/`pitch_shift` (task08).
- Transform-layer wrappers (`AddNoise`, `Speed`) — later progress.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — parent; D2 (this op-cluster extends
   `_functional.hpp`), D5 (testing/coverage), D6 (task07 is independent, any order).
2. torchaudio v2.5.1 source (authoritative signatures/validation/algorithm):
   `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/functional.py`
   (functions `fftconvolve`, `add_noise`, `speed`, helpers `_check_shape_compatible`,
   `_apply_convolve_mode`).

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — append the 3 ops; reuse
  `_apply_convolve_mode` (line ~17), the `convolve_mode` enum (line ~15), the `convolve` shape-align /
  broadcast logic (lines ~47-95) as the model for `fftconvolve`, and `resample` (line ~421) which
  `speed` calls.
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` — only if an option struct
  is warranted (see Notes: none expected; all params are plain args).
- `unit_test/audio/functional/main.cpp` — add tests + register in `main()`'s call list.
- `unit_test/audio/functional/gen_golden.py` — append golden generators for the cross-checks.

## Deliverables
- `inline auto fftconvolve(tensor_t x, tensor_t y, const convolve_mode mode = full) -> tensor_t;` in
  `_functional.hpp`. Same shape-align + broadcast prelude as `convolve` (raise on 0-D; align ndim;
  record original last-dim lengths for the crop); FFT body; final `_apply_convolve_mode` crop using
  the **original** `x`/`y` last-dim lengths.
- `inline auto add_noise(tensor_t waveform, tensor_t noise, tensor_t snr, c10::optional<tensor_t> lengths = c10::nullopt) -> tensor_t;`
  in `_functional.hpp`.
- `inline auto speed(const_tensor_lref_t waveform, int orig_freq, double factor, c10::optional<tensor_t> lengths = c10::nullopt) -> std::tuple<torch::Tensor, c10::optional<torch::Tensor>>;`
  in `_functional.hpp`.
- No new option struct expected (all parameters are scalars/tensors). If one proves convenient,
  follow the `xxx_option` + fluent-setter convention in `_functional_methods_options.hpp`.
- Tests `test_fftconvolve_*`, `test_add_noise_*`, `test_speed_*` in `main.cpp`, registered in
  `main()`; new golden constants in `gen_golden.py` baked into `main.cpp`.

## Steps
1. **fftconvolve** — Reuse the `convolve` prelude: raise `std::invalid_argument` via
   `handle_exceptions<torch::Tensor, std::invalid_argument>(...)` if either input is 0-D; align ndim
   by `unsqueeze(0)` on the shorter-rank operand (torchaudio requires equal ndim — match
   `_check_shape_compatible`); record `original_x_size = x.size(-1)`, `original_y_size = y.size(-1)`.
   Compute `n = x.size(-1) + y.size(-1) - 1`; `auto f = torch::fft::rfft(x, n) * torch::fft::rfft(y, n);`
   `auto result = torch::fft::irfft(f, n);` (both along the default last dim — leading dims broadcast
   in the complex multiply). Return `_apply_convolve_mode(result, original_x_size, original_y_size, mode)`.
   Do NOT flip a kernel here (FFT path is true convolution already).
2. **add_noise validation** — Let `L = waveform.size(-1)`. Match torchaudio's checks exactly:
   require `waveform.dim()-1 == noise.dim()-1 == snr.dim()` and (when `lengths` is set)
   `lengths->dim() == snr.dim()`, else raise `std::invalid_argument` "Input leading dimensions don't
   match."; require `L == noise.size(-1)` else raise "Length dimensions of waveform and noise don't
   match." (include both sizes in the message, like upstream). Use `handle_exceptions<...>` /
   `TORCH_CHECK` consistent with the file.
3. **add_noise energy + scale** — If `lengths` set: build
   `mask = torch::arange(0, L, opts).expand(waveform.sizes()) < lengths->unsqueeze(-1);`
   `masked_waveform = waveform * mask; masked_noise = noise * mask;` else use the originals.
   `energy_signal = torch::linalg::vector_norm(masked_waveform, 2, /*dim=*/-1).pow(2);` (same for
   noise). `original_snr_db = 10 * (torch::log10(energy_signal) - torch::log10(energy_noise));`
   `scale = torch::pow(10.0, (original_snr_db - snr) / 20.0);`
   return `waveform + scale.unsqueeze(-1) * noise` — note the **unmasked** `noise` is added, the mask
   is only for the energy estimate.
4. **speed** — `int src = int(factor * orig_freq); int tgt = int(orig_freq);`
   `int g = std::gcd(src, tgt); src /= g; tgt /= g;`. Compute
   `out_lengths = lengths ? c10::optional(torch::ceil((*lengths) * tgt / src).to(lengths->dtype())) : c10::nullopt;`
   Return `std::make_tuple(resample(waveform, src, tgt), out_lengths)`. (Delegates entirely to the
   ported `resample` for the audio.)
5. **Tests + golden + coverage** — In `main.cpp`:
   - `fftconvolve` vs the ported `convolve`: random `x`,`y`, all three modes, `TM_CHECK_TENSOR_CLOSE`
     to FP tolerance (e.g. atol 1e-4, rtol 1e-4); plus a tiny closed-form case
     (`[1,2,3]` ⊛ `[1,1]` full = `[1,3,5,3]`); a 0-D raise case; a `valid`/`same` length check.
   - `add_noise` closed-form: construct `waveform`,`noise` with known energies (e.g. unit vectors so
     `E=1`) and a chosen `snr`; verify the empirical SNR of the added component equals the requested
     `snr` (recompute `10*log10(E_sig/E_added_noise)` on the output minus waveform). Add a
     `lengths`-masking case (different valid lengths per row change the scale) and the two raise paths.
   - `speed`: assert `std::get<0>(speed(w, of, factor)) == resample(w, src, tgt)` exactly (same call),
     and the `out_lengths` arithmetic (`ceil(lengths*tgt/src)`, dtype preserved) for a `lengths` case
     and `c10::nullopt` for the no-`lengths` case.
   - Extend `gen_golden.py` with `F.fftconvolve`, `F.add_noise`, `F.speed` on small fixed inputs;
     run `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`
     and bake the printed constants into `main.cpp` (no runtime `.venv` dependency).
   - Register every new `test_*` in `main()`. Build & run:
     `cmake --build build --target audio_test_functional && ./build/unit_test/audio/functional/audio_test_functional`;
     `ctest --test-dir build` green; confirm 100% line coverage of the 3 new functions (all branches:
     each `mode`, with/without `lengths`, every raise) under the `TORCHMEDIA_COVERAGE=ON` flow.

## Acceptance criteria
- [ ] `fftconvolve` agrees with the ported `convolve` to FP tolerance for full/valid/same on random
      inputs, and matches the closed-form tiny case and `F.fftconvolve` golden.
- [ ] `add_noise` output's empirical SNR equals the requested `snr` on the closed-form case; the
      `lengths`-masked case matches `F.add_noise` golden; both leading-dim and length-mismatch checks
      raise.
- [ ] `speed` returns `resample(waveform, src, tgt)` for the audio and `out_lengths` =
      `ceil(lengths*tgt/src)` (dtype preserved), `c10::nullopt` when `lengths` is absent; matches
      `F.speed` golden.
- [ ] `ctest --test-dir build` green; 100% line coverage of the 3 new functions in `_functional.hpp`.

## Constraints
- Header-only: `inline` free functions in `torchmedia::audio::functional`; torch-native ops only
  (`torch::fft::rfft`/`irfft`, `torch::linalg::vector_norm`, `torch::arange`/`expand`, `torch::pow`,
  `torch::ceil`, `std::gcd`, `torch::nn::functional::conv1d` via `resample`).
- Match torchaudio's validation/raises exactly (ndim relation, length match, mode set) and its
  semantics: in `add_noise` mask only the energy estimate (add unmasked noise); in `fftconvolve` crop
  uses the **original** input last-dim lengths; in `speed` reduce by `gcd` before resampling.
- `speed` returns `std::tuple<torch::Tensor, c10::optional<torch::Tensor>>`; use `c10::optional`
  (a.k.a. `std::optional`) for the optional `lengths`/`out_lengths`.
- No new heavy deps; reuse `_apply_convolve_mode`, `convolve_mode`, and `resample` already in the file.

## Notes / Assumptions
- Assumption: `resample` (task / commit "Add torch-native resample") is already merged and correct;
  `speed` is a thin wrapper over it and needs no independent resampling logic.
- Assumption: `_apply_convolve_mode`, the `convolve_mode` enum, and `handle_exceptions<T, ExceptionT>`
  are reusable as-is; `fftconvolve` mirrors `convolve`'s shape-align/broadcast prelude (no kernel flip).
- Assumption: `snr` is a per-signal tensor broadcast over time via `scale.unsqueeze(-1)`; tests should
  cover both a scalar-like `snr` (shape matching the leading dims) and a batched case.
- Gotcha: `torch::fft::irfft(..., n)` must be given the explicit length `n` so the real output length
  is `n = x_len + y_len - 1` before cropping; omitting `n` truncates to an even length.
- Gotcha: `add_noise` energy uses `vector_norm(..., ord=2, dim=-1) ** 2` (i.e. sum of squares); do not
  use a plain `norm()` that defaults to a different reduction.
- Gotcha: `speed`'s `out_lengths` must be cast back with `.to(lengths->dtype())` after the `ceil`
  (which produces float) — preserve the caller's integer dtype.
- Question for Mux: none — this task is in-scope (D6 lists task07 as independent, not scope-pending).
