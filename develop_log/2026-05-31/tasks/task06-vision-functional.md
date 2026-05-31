# Task 06 — Vision functional (initial set)
id: 2026-05-31/task06
parent: 2026-05-31/progress03
status: done
owner: code_agent

## Objective
Implement 7 torch-native vision functional ops + a vision test target, each with a torchvision golden
test and 100% coverage.

## Scope
In:
- `hflip, vflip, rgb_to_grayscale, normalize, center_crop, adjust_brightness, invert` in
  `_vision/_functional.hpp`; wire `vision.hpp` to include it.
- `unit_test/vision/functional/{main.cpp,CMakeLists.txt}` + `gen_golden.py`; register in ctest.
Out:
- resize / pad / rotate / color-jitter (Tier 2).

## Inputs (read first, priority order)
1. `develop_log/2026-05-31/progress03-vision-functional-initial.md` — D1/D2.
2. `unit_test/test_util.hpp`.

## Deliverables
- `_vision/_functional.hpp` (7 ops); `vision.hpp` includes it.
- `unit_test/vision/functional/{main.cpp, CMakeLists.txt, gen_golden.py}`; `unit_test/vision/CMakeLists.txt` adds it.

## Steps
1. Implement the 7 ops ([..., C, H, W] float).
2. Add the vision functional test target + golden tests; register with ctest.
3. ctest green; 100% line/branch coverage of `_vision/_functional.hpp`.

## Acceptance criteria
- [x] Each op matches torchvision 0.20.1 golden within tolerance.
- [x] `ctest` green (vision_test_functional + existing targets); 100% coverage of `_vision/_functional.hpp`.

## Constraints
- torch-native (ATen) only; `[..., C, H, W]` float layout; reuse the existing assertion harness.

## Notes / Assumptions
- center_crop assumes crop size <= image size (oversize padding is a Tier-2 follow-up).
