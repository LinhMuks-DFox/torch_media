# Task 03 — Implement create_dct and mfcc
id: 2026-05-31/task03
parent: 2026-05-31/progress02
status: done
owner: code_agent

## Objective
Add `create_dct` (DCT-II matrix) and `mfcc` (DCT over log/dB mel spectrogram) to the functional layer,
each with a torchaudio-golden regression test and 100% coverage.

## Scope
In:
- `create_dct(n_mfcc, n_mels, norm)` -> `[n_mels, n_mfcc]`.
- `mfcc(waveform, mfcc_option)` -> `[..., n_mfcc, time]` via create_dct + melspectrogram + amplitude_to_DB.
- `mfcc_option` struct (sample_rate, n_mfcc, norm, log_mels, embedded mel_spectrogram_option).
- Tests + gen_golden.py entries.
Out:
- griffinlim (task02), resample (task03).

## Inputs
1. `develop_log/2026-05-31/progress02-functional-expansion-tier1.md` — D1/D2.
2. `unit_test/test_util.hpp`.
Code: `_functional.hpp`, `_functional_methods_options.hpp`, `unit_test/audio/functional/main.cpp`.

## Deliverables
- create_dct + mfcc in `_functional.hpp`; `mfcc_option` in the options header.
- Golden tests in `main.cpp`; gen_golden.py updated.

## Steps
1. create_dct: implement + golden test (None/ortho values from D1).
2. mfcc_option + mfcc: implement; golden test against torchaudio MFCC (deterministic input).
3. ctest green; 100% coverage of the new code.

## Acceptance criteria
- [x] create_dct matches torchaudio golden (None & ortho).
- [x] mfcc shape + values match torchaudio MFCC within tolerance (deterministic input).
- [x] `ctest --test-dir build` green; new functions at 100% line/branch coverage.

## Constraints
- torch-native only (ATen ops); follow the existing option-struct + fluent-setter style.

## Notes / Assumptions
- mfcc applies the DCT over the mel axis: `matmul(dct.T, dB_mel)` giving `[n_mfcc, time]`.
