# Progress 03 — Vision functional (initial set)
id: 2026-05-31/progress03
date: 2026-05-31
author: human+ai
status: done
refs: [2026-05-31/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_vision/_functional.hpp
  - libtorchmedia/include/torchmedia/vision.hpp
  - unit_test/vision/functional/{main.cpp,CMakeLists.txt,gen_golden.py}
  - unit_test/vision/CMakeLists.txt

## Goal
Open the vision module with an initial set of torch-native functional ops mirroring
torchvision.transforms.functional. Image tensors follow torchvision's `[..., C, H, W]`, float in [0,1].

## Context / Motivation
Vision was empty stubs. The maintainer is not a vision expert, so this picks a small, high-frequency,
torch-native-clean set (geometry + color + normalization) as the foundation — each verified against
torchvision 0.20.1 golden values (now installed in the .venv) with 100% coverage.

## Decisions

### D1 — Initial vision functional set (7 ops)
- `hflip` / `vflip`: horizontal/vertical flip (`flip(-1)` / `flip(-2)`).
- `rgb_to_grayscale`: L = 0.2989 R + 0.587 G + 0.114 B (num_output_channels 1 or 3). golden [0,0,0]=0.277447.
- `normalize`: (img - mean) / std, per channel. golden (mean=std=0.5) [0,0,0]=-1.0.
- `center_crop(height, width)`: center slice (crop <= image; padding for oversize deferred).
- `adjust_brightness(factor)`: (img*factor).clamp(0,1). golden factor=1.5 [0,0,3]=0.095745.
- `invert`: 1 - img (float bound 1.0). golden [0,0,0]=1.0.
- Layout: `[..., C, H, W]`, float [0,1] (torchvision convention).

### D2 — Test harness reuse + new vision test target
- Reuse `unit_test/test_util.hpp`; add `unit_test/vision/functional/{main.cpp,CMakeLists.txt}`; register ctest.
- Golden from .venv torchvision 0.20.1 (`gen_golden.py`).

## Tasks
- [x] [task06 — vision functional initial set](tasks/task06-vision-functional.md)

## Issues / Gotchas
- torch::tensor(std::vector<double>) builds a double tensor — cast to the image dtype before broadcasting.

## Open / TODO (carry-over)
- Tier 2 vision: resize (interpolate + antialias), pad, rotate/affine, adjust_contrast/saturation/hue.
- mel_filter_bank vectorization (audio, carry-over from progress01/02).

## Agent log
- 2026-05-31 [ai] Installed torchvision 0.20.1 in the .venv; confirmed API + golden for the 7 ops.
  Authored progress03 + task06. Implementing _vision/_functional.hpp next.
- 2026-05-31 [ai] Done: 7 vision ops (hflip/vflip/rgb_to_grayscale/normalize/center_crop/adjust_brightness/
  invert) implemented torch-native in _vision/_functional.hpp; vision.hpp wired; new vision_test_functional
  target. All match torchvision 0.20.1 golden; ctest green (3 tests); _vision/_functional.hpp at 100%
  region/function/line/branch coverage. gen_golden.py added.
