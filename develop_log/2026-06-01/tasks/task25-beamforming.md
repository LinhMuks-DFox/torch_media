# Task 25 — Beamforming (PSD, SoudenMVDR, RTFMVDR, MVDR incl. online)
id: 2026-06-01/task25
parent: 2026-06-01/progress02
status: done              # active | blocked | done
owner: code_agent

## Result
Done 2026-06-01. NEW `_transform_beamform.hpp` (includes `_functional_beamforming.hpp`; aggregated by
`_transform.hpp`): `PSD` (multi_mask channel-average + `functional::psd`, eps default 1e-15),
`SoudenMVDR`/`RTFMVDR` (thin: weights → `apply_beamforming`), and the full `MVDR` (D4) — `mvdr_solution`
enum {ref_channel, stv_evd, stv_power} dispatch, one-hot reference vector built over the leading
(..., channel) dims, cfloat→cdouble promotion with cast-back (G7), and the **online recursive PSD
accumulation** with 4 mutable members (`_psd_s/_psd_n/_mask_sum_s/_mask_sum_n`) + a `bool _initialized`
first-call sentinel (G2 — `MVDR::operator()` is non-const). Tests use **exact** delegation-equivalence:
transform output == the same functional ops replicated in-test (deterministic, so bit-exact — no
phase-invariance needed since it's not compared against a second implementation). Covers PSD
(mask/no-mask/multi_mask), Souden/RTF MVDR, MVDR offline (all 3 solutions + mask_n default + cfloat
cast-back), MVDR online (frame-1 == offline, frame-2 == manual recursion replication, multi_mask path),
ndim<3 / non-complex raises, and every option setter. `audio_test_transform` 151/151; `ctest` 5/5;
`_transform_beamform.hpp` 100% line/region/function/branch coverage. clang-format clean.

## Objective
Implement the multi-channel beamforming transforms: `PSD`, the two thin MVDR wrappers
(`SoudenMVDR`, `RTFMVDR`), and the full `MVDR` class — **including the online recursive PSD
accumulation** (progress02 D4). `MVDR` is the heaviest class of the milestone and the only one with
mutable cross-call state.

## Scope
In:
- `PSD(multi_mask=false, normalize=true, eps=1e-15)` — `operator()(specgram, mask=nullopt)`; when
  `multi_mask`, average the mask over the channel dim (`mask.mean(-3)`) before
  `functional::psd(specgram, mask, normalize, eps)`.
- `SoudenMVDR()` — `operator()(specgram, psd_s, psd_n, reference_channel, diagonal_loading=true,
  diag_eps=1e-7, eps=1e-8)` → `functional::mvdr_weights_souden` then
  `functional::apply_beamforming`.
- `RTFMVDR()` — `operator()(specgram, rtf, psd_n, reference_channel, ...)` →
  `functional::mvdr_weights_rtf` then `functional::apply_beamforming`.
- `MVDR(ref_channel=0, solution=ref_channel, multi_mask=false, diag_loading=true, diag_eps=1e-7,
  online=false)` — full faithful port (D4): solution dispatch, one-hot reference vector,
  cfloat→cdouble promotion (G7), and the **online recursive PSD accumulation** with 4 mutable
  accumulator members + a `bool initialized_` first-call sentinel (G2).
Out:
- Everything non-beamforming.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress02-...md` — D4 (full online port), D5 (`mvdr_solution` enum),
   G2 (non-const `operator()`, mutable accumulators), G7 (dtype promotion).
2. `libtorchmedia/include/torchmedia/_audio/_functional_beamforming.hpp` — `psd`,
   `mvdr_weights_souden` (int + tensor overloads), `mvdr_weights_rtf`, `rtf_evd`, `rtf_power`,
   `apply_beamforming` signatures.
3. torchaudio v2.5.1 `transforms/_multi_channel.py`: `PSD`, `SoudenMVDR`, `RTFMVDR`, `MVDR` and the
   module-level helpers `_get_mvdr_vector`, `_get_updated_mvdr_vector`, `_get_updated_psd_speech`,
   `_get_updated_psd_noise` (the online recursion to mirror exactly).

Code to inspect/change:
- NEW `_transform_beamform.hpp` (per D6) for the four classes + the online helpers.
- `_transform.hpp`, `unit_test/audio/transform/{main.cpp, gen_golden.py}`.

## Deliverables
- The four classes in `torchmedia::audio::transform`:
  - `PSD{ bool multi_mask_, normalize_; double eps_; operator()(spec, mask=nullopt) const; }`.
  - `SoudenMVDR`/`RTFMVDR` — stateless; `operator()` computes weights then applies beamforming.
  - `MVDR` — `mvdr_solution` enum {ref_channel, stv_evd, stv_power}; holds a `PSD psd_`; mutable
    accumulators `psd_s_`, `psd_n_`, `mask_sum_s_`, `mask_sum_n_` and `bool initialized_`;
    **non-const** `operator()(specgram, mask_s, mask_n=nullopt)`.
- Online and offline tests; golden cross-checks (phase-invariant quantities where eigenvectors are
  ambiguous); golden block in `gen_golden.py`.

## Steps
1. **`PSD`** — `operator()`: if `mask` given and `multi_mask_`, `mask = mask.mean(-3)`; return
   `functional::psd(spec, mask, normalize_, eps_)`.
2. **`SoudenMVDR`/`RTFMVDR`** — `operator()`: compute weights via the matching functional op, then
   `functional::apply_beamforming(weights, specgram)`. Thin; use the appropriate
   reference_channel overload (int or one-hot tensor).
3. **`MVDR` offline first** — `_get_mvdr_vector(psd_s, psd_n, ref_vec)` dispatch:
   - `ref_channel` → `functional::mvdr_weights_souden`.
   - `stv_evd` → `functional::rtf_evd(psd_s)` then `functional::mvdr_weights_rtf`.
   - `stv_power` → `functional::rtf_power(psd_s, psd_n, ref)` then `functional::mvdr_weights_rtf`.
   `operator()`: validate `specgram.ndim>=3` and complex; promote cfloat→cdouble (G7); build PSDs
   via `psd_(spec, mask_s)`/`psd_(spec, mask_n)` (with `mask_n = mask_n ? *mask_n : 1 - mask_s`);
   build one-hot `u` over `shape[:-2]` with `u[..., ref_channel]=1`; weights = `_get_mvdr_vector`;
   `out = functional::apply_beamforming(weights, spec)`; cast back to the original dtype.
4. **`MVDR` online** (D4/G2) — when `online_`: on first call (`!initialized_`) store the PSDs and
   `mask_sum_*`, set `initialized_=true`; on later calls blend recursively:
   `num = mask_sum/(mask_sum + mask.sum(-1)); den = 1/(mask_sum + mask.sum(-1));
   psd = psd_old*num[...,None,None] + psd_new*den[...,None,None]; mask_sum += mask.sum(-1)`
   for speech and noise. Store back into the mutable members. `operator()` is non-const.
5. **Tests + golden + ctest + coverage** — offline: golden vs `T.MVDR(online=False)` / `T.PSD` /
   `T.SoudenMVDR` / `T.RTFMVDR` on a fixed complex multi-channel spectrogram + mask; test all three
   `solution` modes. Online: feed two consecutive frames and assert the accumulated PSD matches
   `T.MVDR(online=True)` over the same sequence. Use phase-invariant comparisons for EVD/power
   solutions (eigenvectors are sign/phase-ambiguous). Coverage 100% incl. each `solution`, the
   online first-call vs blend branch, the `mask_n`-null branch, and the ndim/complex raises.

## Acceptance criteria
- [ ] Four classes exist, `torchmedia::audio::transform`; `PSD`/`SoudenMVDR`/`RTFMVDR` `const`,
      `MVDR::operator()` non-const (mutable accumulators).
- [ ] `MVDR` offline matches `T.MVDR(online=False)` golden for all three `solution` modes
      (phase-invariant where applicable, atol 1e-4).
- [ ] `MVDR` online matches `T.MVDR(online=True)` over a 2+ frame sequence (recursive PSD blend
      correct; first-call sentinel works).
- [ ] cfloat input promoted to cdouble internally and cast back; ndim<3 / non-complex raise.
- [ ] `ctest --test-dir build` green; 100% line coverage of `_transform_beamform.hpp`.

## Constraints
- Header-only, ATen only — complex `linalg::solve`/`eigh` via the existing functional beamforming
  ops; do not add new linear-algebra. No new .cpp.
- `MVDR` must replicate torchaudio's online recursion exactly (D4); accumulators are members.
- `mvdr_solution` via enum (D5), not a runtime string.
- Keep `clang-format` clean.

## Notes / Assumptions
- Assumption: the functional beamforming ops (`psd`, `mvdr_weights_*`, `rtf_*`,
  `apply_beamforming`) are complete and golden-checked (progress01 task14) — reuse as-is.
- Assumption: eigenvector phase ambiguity means EVD/power-solution golden checks compare magnitudes
  or beamformed-output power, not raw complex weights.
- Dependency: independent of task19–24.
- Question for Mux: confirm `online` is a constructor option (default false) — i.e. a given `MVDR`
  instance is either streaming or not for its lifetime, matching torchaudio.
