# Task 04 — Implement griffinlim
id: 2026-05-31/task04
parent: 2026-05-31/progress02
status: done
owner: code_agent

## Objective
Add `griffinlim` (iterative phase reconstruction from a power spectrogram) using torch::istft/stft,
with a torchaudio golden regression test and 100% coverage.

## Scope
In: `griffinlim` + `griffinlim_option`; deterministic golden test + branch coverage.
Out: resample (task05).

## Inputs
1. `develop_log/2026-05-31/progress02-functional-expansion-tier1.md` — D3.

## Deliverables
- `griffinlim` in `_functional.hpp`; `griffinlim_option` in the options header. [done]
- Golden test (rand_init=false) + branch tests (default window, momentum=0, explicit length, rand_init). [done]

## Steps
1. Implement mirroring torchaudio: `magnitude = spec^(1/power)`; iterate istft -> stft -> phase update with
   momentum (`momentum/(1+momentum)`); final istft with optional length. [done]
2. Golden test + branch tests; ctest green; 100% coverage. [done]

## Acceptance criteria
- [x] griffinlim shape/sum/sample match torchaudio (rand_init=false, deterministic).
- [x] ctest green; 100% line/branch coverage of griffinlim.

## Result
Done 2026-05-31. Uses the same ATen istft/stft ops torchaudio calls, so rand_init=false is reproducible
(shape (1,1984), sum 901.71, [0,1000]=-0.03503 match within tolerance).

## Notes / Assumptions
- angles init = ones (complex) or unit-modulus random phase (rand_init=true).
