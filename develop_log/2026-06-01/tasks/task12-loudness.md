# Task 12 — Implement loudness (ITU-R BS.1770 K-weighted gated LKFS)
id: 2026-06-01/task12
parent: 2026-06-01/progress01
status: done              # active | blocked | done
owner: code_agent

## Objective
Add torch-native `loudness(waveform, sample_rate)` to `_functional.hpp`, computing the ITU-R
BS.1770-4 K-weighted, two-stage-gated integrated loudness (LKFS) exactly as torchaudio v2.5.1.

## Scope
In:
- `loudness` — free function `inline auto loudness(tensor_t waveform, double sample_rate) -> tensor_t`
  in `torchmedia::audio::functional` (returns a 0-dim/scalar tensor of LKFS, one value per batch).
- The full BS.1770 pipeline: K-weighting (treble + highpass biquads), 400ms / 75%-overlap gating
  blocks, channel-weighted block energy, absolute gate (-70 LKFS), relative gate (-10 dB), and the
  doubly-gated integrated loudness.
- Assertion tests + golden constants from `.venv`.
Out:
- Autograd / backward (forward-only, no custom `Function`).
- A `loudness_option` struct — torchaudio's `loudness` takes only `(waveform, sample_rate)`; no
  options needed (do NOT add one unless a knob appears).
- Any non-BS.1770 loudness metric; per-block loudness as a public return (internal only).
- Resampling / sample-rate conversion (caller supplies the native rate).

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — D3 (lfilter keystone), D6
   (task12 is Tier-1, after task01), the loudness ↔ biquad dependency.
2. `develop_log/2026-06-01/tasks/task01-lfilter-biquad-filtfilt.md` — provides `lfilter`/`biquad`
   (loudness depends on these transitively through the biquads).
3. `develop_log/2026-06-01/tasks/task02-biquad-designers.md` — provides `treble_biquad` and
   `highpass_biquad`, the two K-weighting filters loudness calls directly.
4. torchaudio v2.5.1 source (authoritative algorithm):
   https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/functional.py
   (the `loudness` function — copy its constants and gating logic exactly).

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` (append `loudness`).
- `libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp` (NEW per task01/task02; source
  of `treble_biquad` / `highpass_biquad`) — ensure `_functional.hpp` can see these symbols (include
  order / aggregation in `torchmedia.hpp`).
- `unit_test/audio/functional/main.cpp` (add `test_loudness_*`, register in `main()`).
- `unit_test/audio/functional/gen_golden.py` (append a loudness golden block).

## Deliverables
- `loudness` in `_functional.hpp`: `inline auto loudness(tensor_t waveform, double sample_rate) -> tensor_t`.
  Accepts `(..., channels, time)`; if input is 1-D `(time,)` or 2-D `(channels, time)`, treat as a
  single example (match torchaudio, which works on `(..., C, T)` and reduces over the last two dims).
- Assertion tests in `main.cpp`: at minimum
  `test_loudness_sine_1khz` (1 kHz tone at a known dBFS → near reference LKFS),
  `test_loudness_white_noise_golden` (cross-checked against torchaudio),
  `test_loudness_channel_raise` (>5 channels throws),
  `test_loudness_multichannel` (5-channel weighting g=[1,1,1,1.41,1.41] applied).
- New golden constants in `gen_golden.py` (printed) baked into `main.cpp`.

## Steps
1. **K-weighting** — apply `treble_biquad(waveform, sample_rate, gain=4.0, central_freq=1500.0,
   Q=1/sqrt(2))` then `highpass_biquad(result, sample_rate, cutoff_freq=38.0, Q=0.5)`. Use the exact
   biquad signatures from task02; do NOT re-derive the filter coefficients here.
2. **Validate channels** — let `num_channels = waveform.size(-2)` (after ensuring ≥2 dims; if 1-D,
   unsqueeze a channel dim). If `num_channels > 5`, raise via `handle_exceptions<tensor_t,
   std::invalid_argument>(...)` / `TORCH_CHECK` with a torchaudio-style message ("Only up to 5
   channels are supported.").
3. **Gating blocks** — `gate_samples = (int64_t)std::round(0.4 * sample_rate)`,
   `step = (int64_t)std::round(gate_samples * 0.25)` (75% overlap). Use
   `waveform.unfold(-1, gate_samples, step)` → shape `(..., C, num_blocks, gate_samples)`.
4. **Per-block weighted energy** — block mean energy `energy = mean(block^2, dim=-1)` (the last
   dim, the within-block sample axis); channel weights `g = tensor([1.0,1.0,1.0,1.41,1.41])` sliced
   to `num_channels` and broadcast over the channel dim; `energy_weighted = sum(g * energy, channel
   dim)` (sum over channels, keep the block dim).
5. **Per-block loudness** — `loudness_block = -0.691 + 10*log10(energy_weighted)` (kweight_bias =
   -0.691). Keep `energy_weighted` (linear) alongside for the gated means below.
6. **Absolute gate** — mask blocks with `loudness_block > -70.0` (absolute threshold). Compute the
   mean of `energy_weighted` over the absolute-gate-passing blocks
   (`sum(masked_energy)/count_nonzero(mask)`).
7. **Relative threshold** — `gamma_rel = -0.691 + 10*log10(mean_gated_energy) - 10.0` (relative
   offset -10 dB).
8. **Relative gate + integrate** — keep blocks with `loudness_block > gamma_rel` AND passing the
   absolute gate (doubly-gated); `LKFS = -0.691 + 10*log10(sum(doubly_masked_energy) /
   count_nonzero(doubly_mask))`. Match torchaudio's `count_nonzero` division: if no block passes the
   result is NaN — DO NOT special-case it (parity with torchaudio).
9. **Batching** — reductions in steps 4–8 are over the channel and block dims; any leading batch
   dims survive, yielding one LKFS per batch element. Use `torch::where`/`masked` ops or boolean
   masks with `sum`/`count_nonzero` along the block dim (`dim=-1` after channel reduction) so it
   broadcasts over batch; verify against a stacked-batch torchaudio golden.
10. **Tests + golden + coverage** — add the `test_loudness_*` functions, register them in `main()`;
    extend `gen_golden.py` with the loudness block (1 kHz sine at a known amplitude; a fixed-seed
    white-noise buffer; a 5-channel case) and run
    `/home/mux/code_workspace/torch_media/.venv/bin/python unit_test/audio/functional/gen_golden.py`;
    bake the printed LKFS constants into `main.cpp`. Build & run
    `cmake --build build --target audio_test_functional &&
    ./build/unit_test/audio/functional/audio_test_functional`; `ctest --test-dir build` green; 100%
    line coverage of the new lines in `_functional.hpp` (incl. the >5-channel raise branch).

## Acceptance criteria
- [ ] `loudness` LKFS matches torchaudio v2.5.1 within tolerance (`TM_CHECK_CLOSE`, atol ≈ 1e-2 dB)
      on: a 1 kHz sine at a known amplitude, a fixed-seed white-noise buffer, and a 5-channel buffer.
- [ ] >5 channels throws (`TM_CHECK` on the caught exception); ≤5 channels applies g=[1,1,1,1.41,1.41].
- [ ] A 1 kHz full-scale-relative tone yields an LKFS near the expected reference (sanity, not just
      self-consistency).
- [ ] `ctest --test-dir build` green; 100% line coverage of the new `_functional.hpp` lines.

## Constraints
- Header-only, `inline`, namespace `torchmedia::audio::functional`; torch-native ATen ops only
  (`unfold`, `mean`, `sum`, `log10`, `count_nonzero`, `where`, masking/`index`). No new runtime deps.
- Match torchaudio's validation/raises and numeric constants EXACTLY: kweight_bias `-0.691`,
  absolute gate `-70` LKFS, relative offset `-10` dB, channel weights `[1,1,1,1.41,1.41]`, K-weight
  filters `treble(gain=4dB, f=1500, Q=1/sqrt2)` + `highpass(f=38, Q=0.5)`, block 0.4 s, step 0.25×.
- Reuse `treble_biquad`/`highpass_biquad` from task02 — do not inline new coefficient math (keeps a
  single source of truth and avoids the lfilter recurrence here).
- Preserve torchaudio's NaN-on-empty behavior (`count_nonzero` denominator); do not guard it.

## Notes / Assumptions
- Assumption: task01 (`lfilter`/`biquad`) and task02 (`treble_biquad`/`highpass_biquad`) are merged
  and correct before this task runs (blocked-on, per D6). If task02 lands the biquads in
  `_functional_filtering.hpp`, ensure that header is visible to `_functional.hpp` (include or move
  the call so aggregation order in `torchmedia.hpp` is satisfied). If not yet available, mark this
  task `blocked` and stop.
- Assumption: torchaudio computes the mean within a block over the sample axis (last dim of the
  unfolded tensor); confirm dim order from the source before reducing (unfold appends the window as
  the trailing dim).
- Assumption: golden LKFS values are reproducible from `.venv` torchaudio 2.5.1 on a fixed-seed
  input; bake the printed scalars into `main.cpp` (build must not depend on `.venv` at runtime).
- Question for Mux: none — `loudness` has a fixed `(waveform, sample_rate)` signature with no
  optional knobs, so no `loudness_option` is introduced. Confirm if a fluent option struct is
  nonetheless desired for API symmetry with the other ops (default: do not add one).
