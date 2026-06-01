# Task 22 — Augmentation (masking, SpecAugment, Fade)
id: 2026-06-01/task22
parent: 2026-06-01/progress02
status: done              # active | blocked | done
owner: code_agent

## Result
Done 2026-06-01. NEW `_transform_augment.hpp` (aggregated by `_transform.hpp`): `detail::AxisMasking`
base (eff_axis = `axis + dim - 3`, iid dispatch), `FrequencyMasking` (axis=1), `TimeMasking` (axis=2,
ctor p-validation), `SpecAugment` (mean-or-zero mask value, n_time/n_freq mask accumulation loops,
iid-vs-shared branch on `dim>2 && iid_masks`; no time-warp, matching upstream), and `Fade`
(self-implemented fade-in/out curve synthesis, 5-shape enum, rebuilt per call). Masking delegates to
`functional::mask_along_axis(_iid)`. Tests: Fade golden vs `torchaudio.transforms.Fade` for all 5
shapes + default-no-fade identity; masking/SpecAugment via **seeded delegation-equivalence** (seed →
transform == seed → functional, exact) + mask_param=0 / n_masks=0 identity + TimeMasking p-validation
raise. `audio_test_transform` 100/100; `ctest` 5/5; `_transform_augment.hpp` 100% line/region/function
coverage (4 partial branches: Fade switch-default + SpecAugment `&&` short-circuit combos; line
coverage is the project bar). clang-format clean.

## Objective
Implement the augmentation transforms: the `_AxisMasking` base + `FrequencyMasking`/`TimeMasking`,
`SpecAugment` (mask accumulation, no time-warp — matching upstream), and `Fade` (self-implemented
fade-curve synthesis, the one class here with genuinely new logic and no functional counterpart).

## Scope
In:
- `transform::detail::AxisMasking` (internal base) — axis remap `axis + specgram.dim() - 3`, then
  dispatch `functional::mask_along_axis_iid` vs `functional::mask_along_axis` on `iid_masks`.
- `FrequencyMasking` — `AxisMasking(freq_mask_param, axis=1, iid_masks)`.
- `TimeMasking` — `AxisMasking(time_mask_param, axis=2, iid_masks, p)`; validate `0<=p<=1`.
- `SpecAugment` — `mask_value = zero_masking ? 0.0 : specgram.mean()`; apply `n_time_masks` time
  masks then `n_freq_masks` freq masks in a loop; `iid` branch when `specgram.dim()>2`. No
  time-warping (matches torchaudio 2.5.1).
- `Fade` — `operator()(wav)` = `fade_in_curve(len) * fade_out_curve(len) * wav`, curves built per
  call from `fade_shape` enum {linear, exponential, logarithmic, quarter_sine, half_sine}.
Out:
- Time-domain transforms (task21). Everything else.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress02-...md` — D5 (`fade_shape` enum), G5 (mask RNG lives in the
   functional calls; tests are property/shape).
2. `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — `mask_along_axis`,
   `mask_along_axis_iid` signatures.
3. torchaudio v2.5.1 `transforms/_transforms.py`: `_AxisMasking`, `FrequencyMasking`, `TimeMasking`,
   `SpecAugment`, `Fade` — esp. the exact `_fade_in`/`_fade_out` curve formulas per shape.

Code to inspect/change:
- NEW `_transform_augment.hpp` (per D6) for all five.
- `_transform.hpp`, `unit_test/audio/transform/{main.cpp, gen_golden.py}`.

## Deliverables
- The classes in `torchmedia::audio::transform` (`AxisMasking` in `transform::detail`):
  - `AxisMasking` with `mask_param`, `axis`, `iid_masks`, `p`; `operator()(specgram, mask_value)`.
  - `FrequencyMasking`/`TimeMasking` thin subclasses fixing `axis` (and `p` for time).
  - `SpecAugment` — config ints/bools (`n_time_masks`, `time_mask_param`, `n_freq_masks`,
    `freq_mask_param`, `iid_masks`, `p`, `zero_masking`); `operator()(specgram)`.
  - `Fade` with `fade_in_len`, `fade_out_len`, `fade_shape` enum; `operator()(wav)`.
- Property/shape tests + golden where deterministic; golden block in `gen_golden.py`.

## Steps
1. **`AxisMasking`** — compute `dim = specgram.dim()`, `eff_axis = axis + dim - 3`; branch on
   `iid_masks` to call the matching functional masking fn with `(mask_param, mask_value, eff_axis,
   p)`. RNG is inside the functional call (G5).
2. **`FrequencyMasking`/`TimeMasking`** — construct the base with the fixed axis; `TimeMasking`
   validates `p` in its ctor.
3. **`SpecAugment`** — `mask_value = zero_masking ? 0.0 : specgram.mean().item<double>()`;
   `time_dim=dim-1`, `freq_dim=dim-2`; loop `n_time_masks` then `n_freq_masks`, reassigning
   `specgram` each iteration; pick iid vs shared based on `dim>2 && iid_masks`.
4. **`Fade`** — implement `_fade_in(length, device)` and `_fade_out(length, device)` exactly per the
   torchaudio formulas (linear=identity ramp; exponential=`pow(2, x-1)*x`; logarithmic=
   `log10(0.1+x)+1`; quarter_sine=`sin(x*pi/2)`; half_sine=`sin(x*pi - pi/2)/2 + 0.5`; out-curves
   are the mirror), concatenate with `ones(length - fade_len)`, `clamp(0,1)`; multiply both with the
   waveform. Curves depend on runtime length+device — build in `operator()`, not the ctor.
5. **Tests + golden + ctest + coverage** — `Fade` is deterministic → golden vs `T.Fade` for each
   `fade_shape`. Masking/SpecAugment are RNG → property tests: masked-fraction bounds, shape
   preserved, `mask_value` fill correct, and (with a matched torch seed if feasible) a golden;
   otherwise assert structural properties. Coverage 100% incl. every `fade_shape` branch and the
   iid/non-iid SpecAugment branch.

## Acceptance criteria
- [ ] `AxisMasking` + `FrequencyMasking`/`TimeMasking` dispatch correctly; `TimeMasking` rejects
      `p` outside `[0,1]`.
- [ ] `SpecAugment` applies the right number of time/freq masks, mean-fill vs zero-fill works, iid
      branch taken only for `dim>2`.
- [ ] `Fade` matches baked torchaudio golden for all five `fade_shape` values (atol 1e-6).
- [ ] RNG tests are property/shape-based and stable; `ctest --test-dir build` green.
- [ ] 100% line coverage of `_transform_augment.hpp`.

## Constraints
- Header-only, ATen only (`linspace`, `pow`, `log10`, `sin`, `cat`, `clamp`, `mean`). No new .cpp.
- `SpecAugment` must NOT add time-warping (match torchaudio 2.5.1).
- `Fade` curves rebuilt per call (depend on input length); do not cache.
- Keep `clang-format` clean.

## Notes / Assumptions
- Assumption: an unknown `fade_shape` should be made unrepresentable via the enum (torchaudio
  silently falls through to the raw ramp — we improve on that with a closed enum).
- Assumption: `SpecAugment.mask_value` uses the input mean computed at runtime; replicate exactly.
- Dependency: independent of all other tasks (functional masking already exists).
- Question for Mux: seeding policy for the masking RNG (shared with task21's SpeedPerturbation) —
  thread an optional generator/seed through, or rely on global torch RNG?
