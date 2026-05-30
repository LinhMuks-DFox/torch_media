# Task 02 — Red-test then fix the 5 confirmed functional bugs
id: 2026-05-31/task02
parent: 2026-05-31/progress01
status: done
owner: code_agent

## Objective
For each of the 5 confirmed bugs, add a failing assertion test first, then fix `_functional.hpp` until green.

## Scope
In:
- A test case + fix for each: (1) mel Slaney scale, (2) amplitude_to_DB semantics, (3) db_to_amplitude formula,
  (4) convolve same/valid length, (5) spectrogram double-normalization.
- `amplitude_to_db_option` gains a `multiplier` field.
Out:
- Tier-1 expansion (mfcc / griffinlim) — later progress.
- `mel_filter_bank` vectorization (perf, C-class) — follow-up.

## Inputs (read first, priority order)
1. `develop_log/2026-05-31/progress01-functional-correctness-and-test-harness.md` — D1 lists each bug + correct semantics.
2. `unit_test/test_util.hpp` (from task01).

Code to change:
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp`, `_functional_methods_options.hpp`.

## Deliverables
- A test case per bug in `unit_test/audio/functional/main.cpp` (golden = closed-form / libtorch self-reference;
  cross-checked against `.venv` torchaudio during development).
- Fixed `_functional.hpp` + option struct.

## Steps (per bug: red -> fix -> green)
1. `db_to_amplitude`: assert `ref·10^(0.1·x·power)` on known constants; fix formula.
2. `amplitude_to_DB`: assert `multiplier·log10(clamp) - multiplier·db_multiplier`; remove `pow(2)`;
   add `multiplier` field; per-sample top_db.
3. `convolve`: assert `same` returns first-input length and `valid` returns `max-min+1`; record original x
   length before swap.
4. `mel_filter_bank`: assert Slaney piecewise values + `htk != slaney`; implement the Slaney scale.
5. `spectrogram`: assert `normalized=true` matches a single normalization (libtorch self-reference);
   pass `normalized=false` into `torch::stft`.

## Acceptance criteria
- [x] One red test per bug existed first and now passes.
- [x] `ctest --test-dir build` green.
- [x] `llvm-cov` shows 100% line coverage for `_audio/_functional.hpp` (excluding `_vendor/`).
- [x] Public function names unchanged (`amplitude_to_db_option` gains `multiplier`, documented).

## Constraints
- Each bug is fixed only after its red test exists.
- No regression in `audio_test_io`.

## Notes / Assumptions
- Where closed-form is impractical (mel/spectrogram), generate a golden tensor from the `.venv` torchaudio
  and compare with `TM_CHECK_TENSOR_CLOSE`.
