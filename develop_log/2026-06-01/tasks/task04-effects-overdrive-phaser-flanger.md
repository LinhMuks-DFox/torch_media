# Task 04 — Modulated-delay effects: overdrive, phaser, flanger
id: 2026-06-01/task04
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Port torchaudio's three SoX-style time-domain effects `overdrive`, `phaser`, `flanger` (plus the
shared `_generate_wave_table` helper) into `_functional_filtering.hpp`, on native libtorch ops, each
with a torchaudio golden cross-check and 100% line coverage.

## Scope
In:
- `_generate_wave_table` helper (internal; `"SINE"`/`"TRIANGLE"` wave, `"INT"`/`"FLOAT"` data,
  `table_size`, `min`, `max`, `phase`, range scaling, int +/-0.5 rounding, modulo wrap).
- `overdrive(waveform, gain=20, colour=20)` — cubic soft-clip waveshaper + leaky-integrator recursion.
- `phaser(waveform, sr, gain_in=0.4, gain_out=0.74, delay_ms=3.0, decay=0.4, mod_speed=0.5, sinusoidal=true)`
  — feedback delay with table-modulated read index.
- `flanger(waveform, sr, delay=0, depth=2, regen=0, width=71, speed=0.5, phase=25, modulation="sinusoidal",
  interpolation="linear")` — circular delay buffer, LFO-modulated, linear/quadratic interpolation.
- Option structs `overdrive_option`, `phaser_option`, `flanger_option` (fluent setters).
- Assertion tests + golden constants for all three (both modulation/interpolation modes) and the helper.

Out:
- Autograd / backward (these are forward-only per-sample DSP loops; no custom `torch::autograd::Function`).
- Any non-default branch beyond the two enumerated modes per knob (still cover SINE+TRIANGLE,
  linear+quadratic — those ARE in scope).
- Other SoX effects (`contrast`/`dcshift`/`gain` are task03; biquads are task01/task02).
- Compressed-codec or FFmpeg paths.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D2 (file split: effects live in
   `_functional_filtering.hpp`), D5 (test/coverage rule), D6 (task04 is independent, any order).
2. torchaudio v2.5.1 source (authoritative algorithm + defaults + raises):
   `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/filtering.py`
   — read `_generate_wave_table`, `_dB2Linear`, `overdrive`, `phaser`, `flanger`.
3. `libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp` — target header; match the
   namespace/style of whatever task01/02/03 have landed (or create the file's effects section).

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp` (append the three fns + helper)
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` (the three option structs)
- `unit_test/audio/functional/main.cpp` (tests + register in `main()`'s call list)
- `unit_test/audio/functional/gen_golden.py` (extend; emit golden constants)

## Deliverables
- In `_functional_filtering.hpp`, namespace `torchmedia::audio::functional`:
  - internal `inline torch::Tensor _generate_wave_table(const std::string& wave_type,
    const std::string& data_type, int64_t table_size, double min, double max, double phase,
    torch::Device device)` (place in a detail/anonymous-friendly spot; keep `inline`).
  - `inline torch::Tensor overdrive(const torch::Tensor& waveform, const overdrive_option& opt = {})`.
  - `inline torch::Tensor phaser(const torch::Tensor& waveform, int64_t sample_rate, const phaser_option& opt = {})`.
  - `inline torch::Tensor flanger(const torch::Tensor& waveform, int64_t sample_rate, const flanger_option& opt = {})`.
- In `_functional_methods_options.hpp`: `overdrive_option { double gain=20, colour=20; }`,
  `phaser_option { double gain_in=0.4, gain_out=0.74, delay_ms=3.0, decay=0.4, mod_speed=0.5; bool sinusoidal=true; }`,
  `flanger_option { double delay=0, depth=2, regen=0, width=71, speed=0.5, phase=25; std::string modulation="sinusoidal", interpolation="linear"; }`
  — each with `snake_case` fluent setters returning `*this` (mirror `spectrogram_option`).
- In `main.cpp`: `test_generate_wave_table`, `test_overdrive`, `test_phaser`, `test_flanger`
  (registered in `main()`); golden constants baked from `gen_golden.py`.

## Steps
1. **`_generate_wave_table` helper** — `phase_offset = (int64_t)(phase / pi / 2 * table_size + 0.5)`;
   `point = (torch::arange(table_size) + phase_offset).remainder(table_size)`.
   SINE branch: `d = (torch::sin(point.to(kFloat64) / table_size * 2 * pi) + 1) / 2` -> range `[0,1]`.
   TRIANGLE branch: `seg = torch::div(4 * point, table_size, /*rounding_mode=*/"floor")`; build the 4
   piecewise segments (rising 0->1, falling 1->-? per SoX) matching torchaudio exactly. Scale:
   `d = d * (max - min) + min`. If `data_type=="INT"`: for `d<0` subtract 0.5 else add 0.5, then
   `.to(kInt32)` (use `torch::where`); if `"FLOAT"`: `.to(kFloat32)`. Return.
2. **`overdrive`** — `_dB2Linear(gain) = std::exp(gain * std::log(10.0) / 20.0)`; `colour /= 200`;
   `actual_shape = waveform.sizes()`; reshape to 2D `(n, time)`. `temp = waveform * gain + colour`;
   cubic soft-clip with `torch::where`/`clamp`: `temp<-1 -> -2/3`, `temp>1 -> 2/3`,
   else `temp - temp.pow(3)/3`. Allocate `output`, `last_in = zeros(n)`, `last_out = zeros(n)`;
   sequential loop `for i in time`: `last_out = temp[:,i] - last_in + 0.995*last_out;
   last_in = temp[:,i]; output[:,i] = waveform[:,i]*0.5 + last_out*0.75`. Reshape back; `clamp(-1,1)`.
   Use tensor-column ops or `accessor<...>` — keep it explicit and per-sample.
3. **`phaser`** — `delay_buf_len = (int64_t)(delay_ms*1e-3*sample_rate + 0.5)`;
   `mod_buf_len = (int64_t)(sample_rate / mod_speed + 0.5)`. Build int mod table via
   `_generate_wave_table("SINE"/"TRIANGLE", "INT", mod_buf_len, 1.0, (double)delay_buf_len, pi/2)`
   selected by `opt.sinusoidal`. Reshape 2D; `in = waveform * gain_in`; `delay_buf = zeros`.
   Per-sample loop with circular `delay_pos`/`mod_pos`:
   `idx = (delay_pos + mod_buf[mod_pos]) % delay_buf_len; temp = in[:,i] + delay_buf[idx];
   delay_buf[delay_pos] = temp*decay; output[:,i]=temp`; advance + wrap both positions.
   `output *= gain_out`; reshape back; `clamp(-1,1)`.
4. **`flanger`** — validate: `modulation in {"sinusoidal","triangular"}` else raise (use
   `handle_exceptions`/`TORCH_CHECK` with torchaudio's message), `interpolation in {"linear","quadratic"}`
   else raise, and `channels <= 4` else "Max 4 channels allowed". Reshape to 3D `(batch, ch, time)`.
   Normalize: `feedback_gain=regen/100, delay_gain=width/100, channel_phase=phase/100,
   delay_min=delay/1000, delay_depth=depth/1000`; `lfo_length = (int64_t)(sample_rate/speed)`;
   build FLOAT LFO via `_generate_wave_table(..., "FLOAT", lfo_length, delay_min*sr, (delay_min+delay_depth)*sr, 3*pi/2)`.
   Allocate circular `delay_bufs` sized `max_delay+2`. Per-sample loop: per channel compute
   `cur_channel_phase = (int64_t)(ch * lfo_length * channel_phase + 0.5)`, read modulated `delay_tap`,
   split `int_delay`/`frac_delay`; LINEAR (2-tap): `delayed = d0 + (d1-d0)*frac`; QUADRATIC (3-tap):
   compute `a,b` from `d0,d1,d2` then `delayed = d0 + (a*frac + b)*frac`. Apply feedback into the
   buffer and mix to output; advance LFO/buffer positions with wrap. Reshape back; `clamp(-1,1)`.
5. **Options** — add the three `xxx_option` structs in `_functional_methods_options.hpp` with the
   exact defaults above and fluent `set_*`/named setters; thread them through the three signatures.
6. **Tests + golden + coverage** — extend `gen_golden.py`: emit (a) `_generate_wave_table` outputs for
   SINE+INT and TRIANGLE+FLOAT (small `table_size`, nonzero `min`/`max`/`phase`) to verify exact int
   rounding and modulo; (b) `overdrive`/`phaser`/`flanger` on a short multi-channel signal
   (e.g. 2 ch x ~64 samples, sr 8000), `phaser` with `sinusoidal` true AND false, `flanger` covering
   {sinusoidal,triangular} x {linear,quadratic}; print C++ constant arrays. Run
   `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`,
   bake constants into `main.cpp`. Add `test_*` functions using `TM_CHECK`/`TM_CHECK_TENSOR_CLOSE`,
   plus a `TM_CHECK` for each `flanger` raise (bad modulation, bad interpolation, >4 channels).
   Register in `main()`. Build & run:
   `cmake --build build --target audio_test_functional && ./build/unit_test/audio/functional/audio_test_functional`;
   `ctest --test-dir build` green; confirm 100% line coverage of the new lines in `_functional_filtering.hpp`.

## Acceptance criteria
- [ ] `_generate_wave_table` matches torchaudio **exactly** for SINE+INT and TRIANGLE+FLOAT (integer
      table entries bit-exact; float within atol 1e-6).
- [ ] `overdrive`, `phaser`, `flanger` match torchaudio within `TM_CHECK_TENSOR_CLOSE` (atol ~1e-4)
      on the named multi-channel cases, covering `phaser` sinusoidal true/false and `flanger`
      {sinusoidal,triangular} x {linear,quadratic}.
- [ ] `flanger` raises on invalid `modulation`, invalid `interpolation`, and >4 channels (asserted).
- [ ] Output shapes equal input shapes (round-trip through the 2D/3D reshape); outputs `clamp`ed to [-1,1].
- [ ] `ctest --test-dir build` green; 100% line coverage of the new lines.

## Constraints
- Header-only, `inline` free functions in `_functional_filtering.hpp`; torch-native ATen ops only.
- The three time loops are **per-sample sequential** (leaky integrator / feedback delay) — NOT
  vectorizable; write them explicitly (column-wise tensor ops or `accessor<float,N>`), keep on CPU.
- Match torchaudio's validation messages/raises verbatim (flanger string + channel checks).
- Integer table rounding uses the asymmetric +/-0.5 trick (sign-dependent) before the int32 cast —
  do NOT use plain truncation/`round`.
- Do not add a runtime `.venv` dependency; golden values are baked into `main.cpp`.

## Notes / Assumptions
- Assumption: task04 is independent of task01's `lfilter` — these effects do not call `lfilter`/biquad,
  so this task can proceed before/in parallel with task01.
- Assumption: helper, options, and tests live alongside whatever task01/02/03 land; if
  `_functional_filtering.hpp` does not yet exist, create it with the standard header guard + namespace
  and an effects section (coordinate include in `torchmedia.hpp` per D2).
- Assumption: torchaudio's wave_type spelling is `"TRIANGLE"`/`"SINE"` and data_type `"INT"`/`"FLOAT"`;
  verify the exact triangle piecewise math against the source (step 1) before baking goldens — the SoX
  triangle has a specific 4-segment shape, not a naive abs-ramp.
- Question for Mux: keep `_generate_wave_table` strictly internal (anonymous/`_detail` namespace), or
  expose it publicly? Default assumption: internal helper, not part of the public functional surface.
