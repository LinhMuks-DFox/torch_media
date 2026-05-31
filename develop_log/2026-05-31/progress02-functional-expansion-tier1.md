# Progress 02 — Functional expansion (Tier 1): create_dct, mfcc, griffinlim, resample
id: 2026-05-31/progress02
date: 2026-05-31
author: human+ai
status: done
refs: [2026-05-31/progress01]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/_functional.hpp
  - libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp
  - unit_test/audio/functional/main.cpp, unit_test/audio/functional/gen_golden.py

## Goal
Add four torch-native functional ops mirroring torchaudio: `create_dct`, `mfcc`, `griffinlim`, `resample`.
Each lands with a torchaudio-golden regression test and 100% coverage (per the testing policy).

## Context / Motivation
The audio functional layer is now correct (progress01). These Tier-1 ops are the natural next layer:
mfcc exercises the full mel/db/spectrogram chain (regression amplifier), griffinlim verifies the
spectrogram inverse, resample realizes the "loader reports native rate; resampling is a separate
torch-native op" decision from the dr_wav I/O work (2026-05-30/progress01 D3).

## Decisions

### D1 — create_dct(n_mfcc, n_mels, norm="ortho") -> [n_mels, n_mfcc]
- DCT-II matrix: `dct[n,k] = cos(pi/n_mels * (n+0.5) * k)`; `norm==""` -> `*2`; `norm=="ortho"` ->
  column 0 `*= 1/sqrt(2)`, then `*= sqrt(2/n_mels)`.
- Golden (n_mfcc=4, n_mels=8): None sum=16, [0,0]=2, [3,2]=-1.847759; ortho sum=2.828427, [0,0]=0.353553.
- Dependencies: none.

### D2 — mfcc(waveform, mfcc_option) -> [..., n_mfcc, time]
- `mfcc = matmul(create_dct(n_mfcc, n_mels, norm).T, S)` where `S = amplitude_to_DB(melspectrogram)`
  (or `log(mel + 1e-6)` when `log_mels=true`). i.e. apply DCT over the mel axis.
- `mfcc_option`: `sample_rate, n_mfcc=40, norm="ortho", log_mels=false` + an embedded `mel_spectrogram_option`.
- Dependencies: create_dct, melspectrogram, amplitude_to_DB.

### D3 — griffinlim(specgram, griffinlim_option) -> waveform
- Iterative phase reconstruction from a (power) spectrogram using `torch::istft`/`torch::stft`, with
  momentum (torchaudio): `magnitude = spec.pow(1/power)`; iterate `inverse=istft(mag*angles)`,
  `rebuilt=stft(inverse)`, `angles = (rebuilt - momentum/(1+momentum)*tprev)`, normalized to unit modulus.
- `griffinlim_option`: `n_fft, hop_length, win_length, window, power=2, n_iter=32, momentum=0.99,
  length=-1, rand_init=false`.
- Dependencies: torch::istft/stft (libtorch primitives). rand_init=false -> deterministic golden.

### D4 — resample(waveform, orig_freq, new_freq, resample_option) -> waveform
- Band-limited sinc resampling expressed as a conv1d, matching torchaudio's kernel construction
  (`lowpass_filter_width=6`, `rolloff=0.99`, Hann-windowed sinc). gcd-reduce orig/new for the polyphase rate.
- `resample_option`: `lowpass_filter_width=6, rolloff=0.99` (+ method later).
- Dependencies: convolve / conv1d. Realizes I/O decision D3 (no resampling in the loader).

## Tasks
- [x] [task03 — create_dct + mfcc](tasks/task03-create-dct-mfcc.md)
- [x] [task04 — griffinlim](tasks/task04-griffinlim.md)
- [x] [task05 — resample](tasks/task05-resample.md)

## Issues / Gotchas
- griffinlim / resample point-wise golden may differ slightly across implementations (stft/istft and
  sinc-kernel details); use a tolerance or self-consistency where exact match is impractical.

## Open / TODO (carry-over)
- mel_filter_bank vectorization (perf, from progress01) still pending.

## Agent log
- 2026-05-31 [ai] Confirmed torchaudio signatures + golden for the four ops via the .venv (create_dct,
  griffinlim, resample, MFCC). Authored progress02 + task03/04/05 (this day's progress01 + task01/02 are
  the functional fixes; expansion is progress02 + task03/04/05). Implementing create_dct first.
- 2026-05-31 [ai] Done: create_dct, mfcc (+log_mels), griffinlim (+rand_init/momentum/length branches),
  resample all implemented torch-native, each verified against torchaudio golden values from the .venv.
  ctest green; _functional.hpp + options at 100% region/function/line/branch coverage. gen_golden.py updated.
