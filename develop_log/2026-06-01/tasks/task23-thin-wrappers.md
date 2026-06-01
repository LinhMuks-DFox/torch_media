# Task 23 — Companding / emphasis / feature thin wrappers + Vol
id: 2026-06-01/task23
parent: 2026-06-01/progress02
status: done              # active | blocked | done
owner: code_agent

## Result
Done 2026-06-01. NEW `_transform_effects.hpp` (includes `_functional.hpp` + `_functional_filtering.hpp`
since `_functional.hpp` does not pull in filtering; aggregated by `_transform.hpp`): `MuLawEncoding`,
`MuLawDecoding`, `Preemphasis`, `Deemphasis`, `ComputeDeltas`, `SlidingWindowCmn`, `Loudness`, `Vad`
(thin config holders forwarding to the matching `functional::` op — `Vad` forwards all 17 params), and
`Vol` (`vol_gain_type` enum {amplitude, power, db}: amplitude=`wav*gain`, db=`functional::gain`,
power=`functional::gain(10·log10(gain))`, then `clamp(-1,1)`; ctor rejects negative gain for
amplitude/power). Tests: delegation-equivalence to functional (exact) for the 8 wrappers; Vol's three
modes vs manual reference + clamp bound + negative-gain raise; all setters (incl. Vad ×17) exercised.
`audio_test_transform` 123/123; `ctest` 5/5; `_transform_effects.hpp` 100% line/region/function
coverage (1 partial branch in Vol's compound validation). clang-format clean. ComputeDeltas `mode`
kept as a pass-through string (no validation in functional; not enum'd).

## Objective
Implement the thin-wrapper transforms (constructor stores config, `operator()` forwards to one
`functional::` call): `MuLawEncoding`, `MuLawDecoding`, `Preemphasis`, `Deemphasis`,
`ComputeDeltas`, `SlidingWindowCmn`, `Loudness`, `Vad` — plus `Vol`, which is slightly more than a
wrapper (gain-mode dispatch + clamp).

## Scope
In:
- `MuLawEncoding(quantization_channels)` → `functional::mu_law_encoding`.
- `MuLawDecoding(quantization_channels)` → `functional::mu_law_decoding`.
- `Preemphasis(coeff=0.97)` → `functional::preemphasis`.
- `Deemphasis(coeff=0.97)` → `functional::deemphasis`.
- `ComputeDeltas(win_length=5, mode="replicate")` → `functional::compute_deltas`.
- `SlidingWindowCmn(cmn_window=600, min_cmn_window=100, center=false, norm_vars=false)` →
  `functional::sliding_window_cmn`.
- `Loudness(sample_rate)` → `functional::loudness`.
- `Vad(sample_rate, ...~17 params)` → `functional::vad` (config holder, forward all params).
- `Vol(gain, gain_type)` — `gain_type` enum {amplitude, power, db}: amplitude→`wav*gain`;
  db→`functional::gain(wav, gain)`; power→`functional::gain(wav, 10*log10(gain))`; then
  `clamp(-1,1)`. Validate `gain>=0` for amplitude/power.
Out:
- Convolution/noise (task24). Beamforming (task25).

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress02-...md` — D5 (`gain_type` enum), D1 (class idiom).
2. `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — `mu_law_encoding`/`decoding`,
   `preemphasis`, `deemphasis`, `compute_deltas`, `sliding_window_cmn`, `loudness` signatures.
3. `libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp` — `gain` signature (for Vol).
4. torchaudio v2.5.1 `transforms/_transforms.py`: the listed classes; `Vol` mode dispatch + clamp;
   `Vad`'s full parameter list.

Code to inspect/change:
- NEW/extended `_transform_effects.hpp` (per D6) for all nine.
- `_transform.hpp`, `unit_test/audio/transform/{main.cpp, gen_golden.py}`.

## Deliverables
- The nine classes in `torchmedia::audio::transform`, each with its own `xxx_option` (or direct
  ctor args for the trivial ones — keep the `xxx_option` form for consistency where there are >1
  params; tiny ones like `Loudness(sample_rate)` may take a scalar ctor arg).
- `Vol` with the `gain_type` enum and the three-branch dispatch + final clamp.
- Golden cross-checks for all nine; golden block in `gen_golden.py`.

## Steps
1. **The eight pass-throughs** — each ctor stores config; each `operator()` forwards to the matching
   `functional::` fn with the stored config. `Vad` forwards its full parameter set by position/name.
2. **`Vol`** — ctor validates `gain>=0` when `gain_type ∈ {amplitude, power}`; `operator()`:
   ```
   switch (gain_type_) {
     case amplitude: out = wav * gain_; break;
     case db:        out = functional::gain(wav, gain_); break;
     case power:     out = functional::gain(wav, 10.0 * std::log10(gain_)); break;
   }
   return torch::clamp(out, -1.0, 1.0);
   ```
3. **Tests + golden + ctest + coverage** — golden vs each `torchaudio.transforms` class. `Vol`:
   test all three modes + the clamp + the negative-gain raise. Coverage 100% incl. each `Vol`
   branch and the `Vad` parameter forwarding.

## Acceptance criteria
- [ ] All nine classes exist, `torchmedia::audio::transform`, header-only, delegating to functional.
- [ ] `Vol` dispatches amplitude/db/power correctly, clamps to [-1,1], rejects negative gain for
      amplitude/power.
- [ ] All nine match baked torchaudio 2.5.1 golden values (atol 1e-5).
- [ ] `ctest --test-dir build` green; 100% line coverage of `_transform_effects.hpp` (the parts in
      this task).

## Constraints
- Header-only, ATen only. No new .cpp. No reimplementation of the underlying ops — delegate.
- `Vol` gain modes via enum (D5), not a runtime string.
- Keep `clang-format` clean.

## Notes / Assumptions
- Assumption: `functional::gain` takes a dB gain (per progress01); `Vol`'s power mode converts the
  power ratio to dB via `10*log10`. Verify the functional `gain` units before wiring.
- Assumption: `Vad`'s `measure_duration` is optional (`std::optional<double>`); forward as-is.
- Dependency: independent of all other tasks.
- Question for Mux: for the trivial single-arg classes (`Loudness`, `MuLaw*`, `Preemphasis`,
  `Deemphasis`), prefer a scalar ctor arg or still an `xxx_option` struct for uniformity?
