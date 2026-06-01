# Task 20 — Mel & cepstral (matmul-projection + compositors)
id: 2026-06-01/task20
parent: 2026-06-01/progress02
status: done              # active | blocked | done
owner: code_agent
refs: [2026-06-01/task19]

## Result
Done 2026-06-01. NEW `_transform_feature.hpp` (includes `_transform_spectral.hpp`; aggregated by
`_transform.hpp`) with `AmplitudeToDB`, `MelScale`, `InverseMelScale`, `MelSpectrogram`, `MFCC`,
`LFCC`. `MelScale` caches `fb_` via `functional::mel_filter_bank` and projects by
`torch::matmul(_fb, spec)` — **rank-preserving** ((..., freq, time) → (..., n_mels, time)), NOT
`functional::mel_scale` (which collapses leading dims into one batch axis; that mismatch was caught
by the torchaudio golden during dev). `InverseMelScale` solves `relu(linalg_lstsq(fb[None], packed,
driver))` with a `lstsq_driver` enum {gels(default),gelsy,gelsd,gelss}, batch pack/unpack, and
n_mels validation (G4). `MelSpectrogram`=compose(Spectrogram, MelScale); `MFCC`=compose(
MelSpectrogram, AmplitudeToDB)+cached DCT; `LFCC`=compose(Spectrogram, AmplitudeToDB)+cached
linear-fbank+DCT (G3 — composes cached sub-transforms, not `functional::melspectrogram`/`mfcc`).
Tests: delegation-equivalence to the golden-verified functional ops (squeezing functional's spurious
batch axis where applicable) + baked `torchaudio.transforms` 2.5.1 golden (incl. a full-rank (9-bin)
config for InverseMelScale since the default `gels` QR driver needs full rank). `audio_test_transform`
57/57; `ctest` 5/5; **100% line/region/function/branch coverage of `_transform_feature.hpp` and
`_transform_spectral.hpp`**. No functional change needed. Default driver confirmed `gels`,
mel_scale `htk`, AmplitudeToDB top_db=None (no clamp) — matching torchaudio.

## Objective
Implement the mel/cepstral transforms: `AmplitudeToDB`, `MelScale`, `InverseMelScale`,
`MelSpectrogram`, `MFCC`, `LFCC`. These add the layer's signature "beyond-wrapping" logic — cached
filterbank/DCT matrices applied by `matmul` *in the call*, an lstsq inverse with no functional
counterpart, and composition of sub-transforms.

## Scope
In:
- `AmplitudeToDB` — scalar-config wrapper over `functional::amplitude_to_DB` (no cached tensor).
  Needed by `MFCC`/`LFCC`.
- `MelScale` — cache `fb = functional::mel_filter_bank(...)`; `operator()(spec)` applies it by
  `matmul` (NOT a functional call): `(spec.transpose(-1,-2) @ fb).transpose(-1,-2)`.
- `InverseMelScale` — cache `fb`; `operator()(mel)` = `relu(linalg::lstsq(fb.T[None], mel,
  driver).solution)` with batch pack/unpack and `n_mels` validation (see G4).
- `MelSpectrogram` — compose `Spectrogram` + `MelScale` (G3: compose, don't call
  `functional::melspectrogram`).
- `MFCC` — compose `MelSpectrogram` + `AmplitudeToDB`, cache `dct_mat = functional::create_dct(...)`,
  apply DCT by `matmul`; `log_mels` branch `log(mel+1e-6)`.
- `LFCC` — compose `Spectrogram` + `AmplitudeToDB`, cache `filter_mat = functional::linear_fbanks(...)`
  + `dct_mat`, apply both by `matmul`.
Out:
- Anything not mel/cepstral. Vision. RNNTLoss.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress02-audio-transform-layer.md` — G3 (compose, don't re-wrap),
   G4 (InverseMelScale lstsq), D5 (enum the `lstsq_driver`).
2. `develop_log/2026-06-01/tasks/task19-...md` — the class idiom + `Spectrogram` it depends on.
3. `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — `amplitude_to_DB`, `mel_filter_bank`,
   `mel_scale`, `create_dct`, `linear_fbanks` signatures; `mfcc_option`/`mel_spectrogram_option`
   field sets in `_functional_methods_options.hpp` (mirror, don't reuse).
4. torchaudio v2.5.1 `transforms/_transforms.py`: `AmplitudeToDB`, `MelScale`, `InverseMelScale`,
   `MelSpectrogram`, `MFCC`, `LFCC` — note all four matmul projections and the lstsq solve.

Code to inspect/change:
- `_transform_feature.hpp` (NEW) for `AmplitudeToDB`, `MFCC`, `LFCC` (and `ComputeDeltas`/
  `SlidingWindowCmn` land in task23); `MelScale`/`InverseMelScale`/`MelSpectrogram` go in
  `_transform_spectral.hpp` (created in task19) or `_transform_feature.hpp` — implementer's call,
  keep include order (Spectrogram before MelSpectrogram).
- `_transform.hpp` — add the new header.
- `unit_test/audio/transform/{main.cpp, gen_golden.py}`.

## Deliverables
- The six classes in `torchmedia::audio::transform`, each with its own `xxx_option`:
  - `AmplitudeToDB(stype, top_db)` — derive `multiplier=10|20`, `amin=1e-10`, `db_multiplier=0`;
    `operator()` → `functional::amplitude_to_DB`.
  - `MelScale` — ctor builds `fb_` via `functional::mel_filter_bank(n_mels, f_min, f_max,
    sample_rate, n_stft, norm, mel_scale)`; `f_min>f_max` raises; `operator()` matmul-projects.
  - `InverseMelScale` — ctor builds `fb_`; `lstsq_driver` enum {gels,gelsy,gelsd,gelss};
    `operator()` validates `mel.size(-2)==n_mels`, packs `view(-1,n_mels,T)`, solves, `relu`,
    unpacks.
  - `MelSpectrogram` — holds a `Spectrogram` (onesided forced true) + a `MelScale` (over
    `n_fft/2+1` stft bins); `operator()` chains them.
  - `MFCC` — holds `MelSpectrogram` + `AmplitudeToDB('power',80)`; `dct_mat_=create_dct(n_mfcc,
    n_mels, norm)`; `dct_type==2` only; `n_mfcc<=n_mels` else raise; `operator()` matmuls the DCT.
  - `LFCC` — holds `Spectrogram` + `AmplitudeToDB('power',80)`; `filter_mat_=linear_fbanks(n_fft/2+1,
    f_min, f_max, n_filter, sample_rate)`; `dct_mat_=create_dct(n_lfcc, n_filter, norm)`;
    `n_lfcc<=n_fft` else raise; `operator()` matmuls filterbank then DCT.
- Golden cross-checks for all six; golden block in `gen_golden.py`.

## Steps
1. **`AmplitudeToDB`** — ctor derives the scalar constants; `operator()` forwards. Trivial; do
   first since MFCC/LFCC compose it.
2. **`MelScale`** — build `fb_` in ctor; matmul projection helper
   `project(x, M) = torch::matmul(x.transpose(-1,-2), M).transpose(-1,-2)` (reuse across MelScale/
   InverseMelScale/MFCC/LFCC — put it in a `transform::detail`).
3. **`InverseMelScale`** — per G4. `solution = torch::linalg::lstsq(fb_.transpose(-1,-2).unsqueeze(0),
   melspec_packed, /*rcond=*/c10::nullopt, driver_string).solution; specgram = torch::relu(solution);`
   then unpack to `shape[:-2] + (n_freq, T)`. Map the `lstsq_driver` enum to the ATen driver string.
4. **`MelSpectrogram`** — construct sub-`Spectrogram` and sub-`MelScale` in the ctor (forward the
   `melkwargs`/`speckwargs` as nested option fields). `operator()` = `mel_scale_(spectrogram_(w))`.
5. **`MFCC`** — `operator()`: `mel = melspectrogram_(w); mel = log_mels ? torch::log(mel+1e-6) :
   amplitude_to_DB_(mel); return project(mel, dct_mat_);` Validate in ctor.
6. **`LFCC`** — `operator()`: `s = spectrogram_(w); s = project(s, filter_mat_); s = log_lf ?
   log(s+1e-6) : amplitude_to_DB_(s); return project(s, dct_mat_);`
7. **Tests + golden + ctest + coverage** — golden vs `T.AmplitudeToDB/MelScale/InverseMelScale/
   MelSpectrogram/MFCC/LFCC`. For `InverseMelScale`, also assert the reconstruction error is small
   on a round-trip `MelScale`→`InverseMelScale` (property test) in addition to the golden. Coverage
   100% incl. each validation/branch (`log_mels` true/false, each `f_min>f_max`/coeff-count raise,
   each `driver`).

## Acceptance criteria
- [ ] Six classes exist, `torchmedia::audio::transform`, cached `fb_`/`dct_mat_`/`filter_mat_`
      where applicable, matmul projections done in `operator()`.
- [ ] `MelSpectrogram`/`MFCC`/`LFCC` compose sub-transforms (G3) — verified by inspection (no call
      to `functional::melspectrogram`/`functional::mfcc`).
- [ ] `InverseMelScale` uses `torch::linalg::lstsq`+`relu`, validates `n_mels`, round-trips with
      small error; each `lstsq_driver` value works.
- [ ] All six match baked torchaudio 2.5.1 golden values (atol 1e-4 for lstsq, 1e-5 elsewhere).
- [ ] `ctest --test-dir build` green; 100% line coverage of the new/extended headers.

## Constraints
- Header-only, ATen only, `torch::linalg::lstsq`/`relu`/`matmul`/`log`. No new .cpp.
- Compose cached sub-transforms — do not re-wrap functional composites (G3).
- `dct_type==2` only (match torchaudio); raise on others.
- Keep `clang-format` clean.

## Notes / Assumptions
- Assumption: `mel_filter_bank` is the existing functional name for `melscale_fbanks` (progress01
  inventory). Use it; if the arg order differs, adapt — do not change functional.
- Assumption: lstsq atol can be looser (1e-4) since drivers differ slightly in conditioning.
- Dependency: BLOCKED-BY task19 (`Spectrogram`). Independent of task21–25.
- Question for Mux: `InverseMelScale` default `driver` — torchaudio defaults to `gelsd`; confirm.
