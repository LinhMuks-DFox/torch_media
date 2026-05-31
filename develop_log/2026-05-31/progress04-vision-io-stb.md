# Progress 04 — Vision I/O (stb_image, header-only)
id: 2026-05-31/progress04
date: 2026-05-31
author: human+ai
status: done
refs: [2026-05-31/progress03]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_vision/_vendor/stb_image.h, stb_image_write.h (vendored)
  - libtorchmedia/include/torchmedia/_vision/_io.hpp
  - libtorchmedia/include/torchmedia/vision.hpp
  - unit_test/vision/io/main.cpp
  - THIRD_PARTY_LICENSES

## Goal
Add vision image I/O using vendored single-header `stb_image` / `stb_image_write` (header-only,
public domain) — the same pattern as dr_wav for audio. `load_image` -> [C,H,W] float [0,1];
`save_image` -> PNG.

## Context / Motivation
`_vision/_io.hpp` was an empty stub. Reading/writing PNG/JPG needs an image codec; stb is the
header-only, zero-build-dep, public-domain choice consistent with the dr_wav decision (no heavy deps).

## Decisions

### D1 — Vendor stb_image + stb_image_write; gate behind TORCHMEDIA_IO_IMPLEMENTATION
- Same ODR strategy as dr_wav: the implementation is emitted in the one TU that defines
  `TORCHMEDIA_IO_IMPLEMENTATION` (which now also defines `STB_IMAGE_IMPLEMENTATION` /
  `STB_IMAGE_WRITE_IMPLEMENTATION`).
- License: stb is public domain / MIT; add to `THIRD_PARTY_LICENSES`.

### D2 — API + layout
- `load_image(path)` -> `[C, H, W]` float in [0,1]. stb returns `[H,W,C]` uint8; convert /255 + permute to CHW.
- `save_image(img, path)` -> bool. `[C,H,W]` float -> `[H,W,C]` uint8 (clamp [0,1], *255, round) -> PNG.

### D3 — Test: round-trip
- save -> load round-trip with tolerance ~1/255 (uint8 quantization). Replace the empty `vision_test_io`
  stub with an assertion-based round-trip test.

## Tasks
- [x] [task07 — vision image I/O via stb_image](tasks/task07-vision-io.md)

## Issues / Gotchas
- stb load gives HWC uint8 row-major; permute to CHW and divide by 255.
- PNG is lossless but float->uint8->float quantizes (~1/255), so the round-trip needs a tolerance.

## Open / TODO (carry-over)
- JPG quality option, load with a desired channel count, 16-bit/HDR — later.
- Tier 2 vision transforms (resize/pad/rotate/color); audio carry-overs.

## Agent log
- 2026-05-31 [ai] Vendored stb_image + stb_image_write (v2.30); implemented load_image ([C,H,W] float [0,1])
  and save_image (PNG) in _vision/_io.hpp, gated behind TORCHMEDIA_IO_IMPLEMENTATION (same macro as dr_wav);
  wired vision.hpp; replaced the empty vision_test_io stub with a round-trip + branch test; added stb to
  THIRD_PARTY_LICENSES. ctest green (4 tests); _vision/_io.hpp at 100% region/function/line/branch coverage.
