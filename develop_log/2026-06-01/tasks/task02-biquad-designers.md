# Task 02 — Implement the 11 biquad filter designers
id: 2026-06-01/task02
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Append the 11 torchaudio biquad coefficient designers (`allpass_biquad`, `lowpass_biquad`,
`highpass_biquad`, `bandpass_biquad`, `bandreject_biquad`, `band_biquad`, `equalizer_biquad`,
`bass_biquad`, `treble_biquad`, `deemph_biquad`, `riaa_biquad`) to
`_audio/_functional_filtering.hpp`, each computing `(b0,b1,b2,a0,a1,a2)` and delegating to the
task01 `biquad`.

## Scope
In:
- The 11 free functions above in `torchmedia::audio::functional`, all returning the filtered tensor
  via `biquad(waveform, b0,b1,b2,a0,a1,a2)` (task01).
- Validation/raises matching torchaudio: `deemph_biquad` rejects sr ∉ {44100, 48000};
  `riaa_biquad` rejects sr ∉ {44100, 48000, 88200, 96000}.
- Assertion tests + any new golden constants for each designer.
Out:
- `lfilter` / `biquad` / `filtfilt` (task01 — these are dependencies, not in scope here).
- Autograd / backward.
- SoX effects (`contrast`, `dcshift`, `gain`, `overdrive`, `phaser`, `flanger`), `vad`,
  `deemphasis`, `loudness` — separate tasks.
- Option structs: torchaudio passes these as plain positional scalars; do **not** introduce
  `xxx_option` structs for the designers (keep parity with torchaudio's signatures).

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D2 (file split → all biquads live
   in `_functional_filtering.hpp`), D3 (`biquad` is the shared primitive), D6 (this is Tier 1, runs
   after task01).
2. `develop_log/2026-06-01/tasks/task01-lfilter-biquad-filtfilt.md` — the exact `biquad` signature
   to call (argument order `b0,b1,b2,a0,a1,a2`).
3. torchaudio v2.5.1 source (authoritative formulas, incl. the `deemph`/`riaa` hardcoded tables):
   `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/filtering.py`
   (load WebFetch via `ToolSearch "select:WebFetch"` if you need to read it).

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp` (append; created by task01)
- `unit_test/audio/functional/main.cpp` (add tests + register in `main()`'s call list)
- `unit_test/audio/functional/gen_golden.py` (extend to print golden coefficients/outputs)
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` (only if you decide an
  option struct is warranted — default is NOT to add one; see Scope Out)

## Deliverables
- In `_functional_filtering.hpp`, 11 `inline` free functions in `torchmedia::audio::functional`
  (`snake_case`, default args mirroring torchaudio):
  - `allpass_biquad(const tensor_t& waveform, int64_t sample_rate, double central_freq, double Q = 0.707)`
  - `lowpass_biquad(const tensor_t&, int64_t sample_rate, double cutoff_freq, double Q = 0.707)`
  - `highpass_biquad(const tensor_t&, int64_t sample_rate, double cutoff_freq, double Q = 0.707)`
  - `bandpass_biquad(const tensor_t&, int64_t sample_rate, double central_freq, double Q = 0.707, bool const_skirt_gain = false)`
  - `bandreject_biquad(const tensor_t&, int64_t sample_rate, double central_freq, double Q = 0.707)`
  - `band_biquad(const tensor_t&, int64_t sample_rate, double central_freq, double Q = 0.707, bool noise = false)`
  - `equalizer_biquad(const tensor_t&, int64_t sample_rate, double center_freq, double gain, double Q = 0.707)`
  - `bass_biquad(const tensor_t&, int64_t sample_rate, double gain, double central_freq = 100.0, double Q = 0.707)`
  - `treble_biquad(const tensor_t&, int64_t sample_rate, double gain, double central_freq = 3000.0, double Q = 0.707)`
  - `deemph_biquad(const tensor_t&, int64_t sample_rate)`
  - `riaa_biquad(const tensor_t&, int64_t sample_rate)`
- Assertion tests in `main.cpp` (one per designer) + their registration in `main()`.
- New golden constants in `gen_golden.py` (printed) baked into `main.cpp` as literals.

## Steps
1. **Cookbook helper** — at the top of the new code compute, per call,
   `w0 = 2*M_PI*freq/sample_rate`, `cos_w0 = std::cos(w0)`, `sin_w0 = std::sin(w0)`,
   `alpha = sin_w0 / (2*Q)` (use `<cmath>`; these are plain `double`s, not tensors). Each designer
   below fills `b0,b1,b2,a0,a1,a2` then `return biquad(waveform, b0,b1,b2,a0,a1,a2);`.
2. **RBJ-cookbook designers** — implement exactly per FUNCTION DATA:
   - `allpass`: `b0=1-alpha; b1=-2cos; b2=1+alpha; a0=1+alpha; a1=-2cos; a2=1-alpha`.
   - `lowpass`: `b0=(1-cos)/2; b1=1-cos; b2=b0;` a = `{1+alpha, -2cos, 1-alpha}`.
   - `highpass`: `b0=(1+cos)/2; b1=-1-cos; b2=b0;` a same as lowpass.
   - `bandpass`: `temp = const_skirt_gain ? sin_w0/2 : alpha; b0=temp; b1=0; b2=-temp;` a same.
   - `bandreject`: `b0=1; b1=-2cos; b2=1;` a same.
   - `equalizer`: `A = std::pow(10.0, gain/40.0); b0=1+alpha*A; b1=-2cos; b2=1-alpha*A;`
     `a0=1+alpha/A; a1=-2cos; a2=1-alpha/A;` (param is `center_freq`).
3. **`band_biquad` (SoX, NOT cookbook)** — `bw = central_freq/Q; a0=1; a2=std::exp(-2*M_PI*bw/sr);`
   `a1 = -4*a2/(1+a2)*cos_w0; b0 = std::sqrt(1 - a1*a1/(4*a2))*(1-a2);`
   if `noise`: `b0 *= std::sqrt(((1+a2)*(1+a2) - a1*a1) * (1-a2)/(1+a2)) / b0;` `b1=b2=0;`.
4. **Shelf designers** — shared temps `A=pow(10,gain/40); temp1=2*sqrt(A)*alpha; temp2=(A-1)*cos;
   temp3=(A+1)*cos;`
   - `bass_biquad` (low-shelf): `b0=A*((A+1)-temp2+temp1); b1=2*A*((A-1)-temp3);
     b2=A*((A+1)-temp2-temp1); a0=(A+1)+temp2+temp1; a1=-2*((A-1)+temp3); a2=(A+1)+temp2-temp1;`
     then **pre-divide all 6 by a0** before calling `biquad` (torchaudio normalizes here).
   - `treble_biquad` (high-shelf): `b0=A*((A+1)+temp2+temp1); b1=-2*A*((A-1)+temp3);
     b2=A*((A+1)+temp2-temp1); a0=(A+1)-temp2+temp1; a1=2*((A-1)-temp3); a2=(A+1)-temp2-temp1;`
     **no pre-divide** (pass raw `a0`).
5. **`deemph_biquad`** — ISO908 CD de-emphasis (high-shelf). `TORCH_CHECK(sr==44100 || sr==48000, ...)`.
   Hardcode per rate: 44100 → `{cf=5283, S=0.4845, gain=-9.477}`; 48000 → `{cf=5356, S=0.479, gain=-9.62}`.
   `A=pow(10,gain/40); w0=2*M_PI*cf/sr; alpha = sin(w0)/2 * std::sqrt((A + 1/A)*(1/S - 1) + 2);` then
   the same treble (high-shelf) coefficient block as step 4 with these `A,w0,alpha`.
6. **`riaa_biquad`** — RIAA EQ. `TORCH_CHECK(sr in {44100,48000,88200,96000}, ...)`. Hardcode the
   per-rate zeros/poles tables from torchaudio; build `b={1, -(z0+z1), z0*z1}` and `a` from poles the
   same way; then normalize to 0 dB at 1 kHz: compute `y = 2*M_PI*1000/sr` and the complex magnitude
   of `H(e^{-jy}) = B(e^{-jy})/A(e^{-jy})` (use `std::complex<double>`, `std::polar`/`std::exp`),
   `g = 1/|H|`, and multiply `b0,b1,b2` by `g`. Mirror torchaudio's table values verbatim.
7. **Tests + golden + green** — add one assertion test per designer to `main.cpp`:
   (a) **closed-form**: for fixed sr/freq/Q, assert the computed `(b0..a2)` equal the exact formulas
   above (`TM_CHECK_CLOSE`); the cleanest way is to filter an impulse `delta = [1,0,0,...]` and check
   the first few output samples / or expose coefficients via a tiny local recompute in the test.
   (b) **torchaudio cross-check**: run each designer on a fixed waveform in `gen_golden.py`, print the
   output samples, bake the literals into `main.cpp`, and `TM_CHECK_TENSOR_CLOSE` against them.
   (c) **raises**: assert `deemph_biquad(.,22050)` and `riaa_biquad(.,32000)` throw (unsupported rate).
   Register every new test in `main()`'s call list (alongside `test_resample()` etc.). Generate golden:
   `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`.
   Build & run: `cmake --build build --target audio_test_functional &&
   ./build/unit_test/audio/functional/audio_test_functional`; then `ctest --test-dir build` green and
   100% line coverage of the new lines in `_functional_filtering.hpp`.

## Acceptance criteria
- [ ] All 11 designers exist in `_functional_filtering.hpp` with the signatures above and delegate to
      task01 `biquad`.
- [ ] Each designer's coefficients/output match the exact closed-form formulas (cookbook / SoX / shelf)
      within tolerance.
- [ ] Each designer's output matches torchaudio 2.5.1 (golden, baked literals) within
      `TM_CHECK_TENSOR_CLOSE` tolerance on the named fixed input.
- [ ] `bass_biquad` pre-divides by `a0`; `treble_biquad` does not (verified by golden parity).
- [ ] `deemph_biquad` raises on sr ∉ {44100,48000}; `riaa_biquad` raises on sr ∉ {44100,48000,88200,96000}.
- [ ] `ctest --test-dir build` green; 100% line coverage of the new lines (vendored `_vendor/` excluded).

## Constraints
- Header-only: all 11 are `inline` free functions in `_functional_filtering.hpp`; no new `.cpp`.
- Torch-native only: the filtering goes through task01 `biquad` (which uses `lfilter`); coefficient
  math is plain scalar `double` / `std::complex<double>` (`<cmath>`, `<complex>`).
- Match torchaudio's validation/raises exactly (use `TORCH_CHECK` / the project's
  `handle_exceptions<...>` helper, consistent with how task01 raises).
- `band_biquad` is SoX 'band', not RBJ cookbook — do not reuse the cookbook `bandpass` coefficients.
- `riaa_biquad` uses a complex-magnitude normalization at 1 kHz; keep it numeric (no symbolic),
  matching torchaudio's per-rate tables verbatim.

## Notes / Assumptions
- **Depends on task01**: requires the merged `biquad(waveform, b0,b1,b2,a0,a1,a2)` in
  `_functional_filtering.hpp`. If task01 is not yet merged, set this task `blocked` and stop.
- Assumption: the designers take plain positional scalars (no `xxx_option` structs), matching
  torchaudio's signatures; defaults (`Q=0.707`, `central_freq=100/3000`, `const_skirt_gain=false`,
  `noise=false`) are taken from torchaudio v2.5.1.
- Assumption: `bass_biquad` pre-divides the 6 coefficients by `a0` and `treble_biquad` does not — this
  asymmetry is real in torchaudio (noted in progress01); the golden cross-check is the guard.
- Assumption: `tensor_t` is the existing project alias for `torch::Tensor` used in the audio headers;
  reuse it (do not introduce a new alias).
- Question for Mux: none — all 11 designers are confirmed in-scope for this task (Tier 1, D6). The
  scope-pending items (D4) are the SoX effects / `vad`, which are separate tasks.
