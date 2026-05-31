# Task 05 — Implement resample
id: 2026-05-31/task05
parent: 2026-05-31/progress02
status: done
owner: code_agent

## Objective
Add torch-native `resample` (band-limited sinc expressed as a strided conv1d) mirroring torchaudio,
with a torchaudio golden regression test and 100% coverage.

## Scope
In: `resample` + `resample_option`; golden tests (downsample + non-integer ratio).
Out: the kaiser-window method (Hann only for now).

## Inputs
1. `develop_log/2026-05-31/progress02-functional-expansion-tier1.md` — D4.

## Deliverables
- `resample` in `_functional.hpp`; `resample_option` in the options header. [done]
- Golden tests (64->32, 64->48). [done]

## Steps
1. Build the `[nf, 1, K]` sinc filterbank (gcd-reduced rate, Hann window, rolloff). [done]
2. Pad + strided conv1d + crop to `ceil(nf*length/of)`. [done]
3. Golden test; ctest green; 100% coverage. [done]

## Acceptance criteria
- [x] resample shape/sum/samples match torchaudio (64->32, 64->48) within tolerance.
- [x] ctest green; 100% line/branch coverage.

## Result
Done 2026-05-31. Mirrors torchaudio.functional.resample (sinc_interp_hann) via the same ATen ops
(arange/cos/sin/where/conv1d), matching point-wise (e.g. 64->32 sum 0.049087, [0,1]=0.546815).

## Notes / Assumptions
- gcd-reduces orig/new; only the Hann-windowed sinc method is implemented.
