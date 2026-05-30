# Task 01 — Assertion test harness + coverage + ctest wiring
id: 2026-05-31/task01
parent: 2026-05-31/progress01
status: done
owner: code_agent

## Objective
Stand up a header-only assertion test harness with LLVM coverage and ctest registration, so functional tests
are assertion-based (not eyeball) and coverage can be measured.

## Scope
In:
- `unit_test/test_util.hpp`: `TM_CHECK` / `TM_CHECK_CLOSE` / `TM_CHECK_TENSOR_CLOSE`; `main` returns a summary code.
- CMake: `TORCHMEDIA_COVERAGE` option; make `test_util.hpp` includable from any test.
- Rewrite `unit_test/audio/functional/main.cpp` to assertion-based smoke tests; register with ctest.
Out:
- The 5 bug red-tests + fixes (task02).
- Splitting into per-function test executables (single executable for now).

## Inputs (read first, priority order)
1. `develop_log/2026-05-31/progress01-functional-correctness-and-test-harness.md` — decisions D2/D3/D4.

Code to inspect/change:
- `unit_test/CMakeLists.txt`, `unit_test/audio/functional/{main.cpp,CMakeLists.txt}`, root `CMakeLists.txt`.

## Deliverables
- `unit_test/test_util.hpp`.
- `TORCHMEDIA_COVERAGE` option in root CMakeLists + `include_directories` so `test_util.hpp` resolves.
- assertion-based `unit_test/audio/functional/main.cpp` + `add_test(audio_test_functional)`.

## Steps
1. Write `test_util.hpp`.
2. Add coverage option + include dir in CMake.
3. Rewrite functional `main.cpp` to `TM_CHECK`-based smoke tests using bug-free assertions (full-convolve
   length = N+M-1; spectrogram freq bins = n_fft/2+1).
4. `add_test(audio_test_functional)`; build; `ctest --test-dir build` green.

## Acceptance criteria
- [x] `ctest --test-dir build` runs `audio_test_functional` and `audio_test_io`, both green.
- [x] A `TM_CHECK` failure makes the executable return non-zero (ctest reports the case as failed).
- [x] `cmake -S . -B build -DTORCHMEDIA_COVERAGE=ON` configures and builds.

## Constraints
- No third-party test library. CPU-only tests (no hard `mps` dependency).

## Notes / Assumptions
- The functional test no longer depends on audio I/O (uses synthetic tensors), so no
  `TORCHMEDIA_IO_IMPLEMENTATION` is needed in that TU.
