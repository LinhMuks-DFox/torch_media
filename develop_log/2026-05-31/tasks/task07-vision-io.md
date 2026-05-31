# Task 07 — Vision image I/O via stb_image
id: 2026-05-31/task07
parent: 2026-05-31/progress04
status: done
owner: code_agent

## Objective
Implement `load_image` / `save_image` in `_vision/_io.hpp` using vendored header-only stb_image /
stb_image_write, with a round-trip test and 100% coverage.

## Scope
In:
- Vendor `stb_image.h` + `stb_image_write.h` into `_vision/_vendor/`.
- `load_image(path) -> [C,H,W] float [0,1]`; `save_image(img, path) -> bool` (PNG).
- Gate the implementation behind `TORCHMEDIA_IO_IMPLEMENTATION`.
- Wire `vision.hpp`; replace the empty `vision_test_io` with a round-trip test; add stb to THIRD_PARTY_LICENSES.
Out:
- JPG quality / desired-channels / 16-bit (Tier 2).

## Inputs
1. `develop_log/2026-05-31/progress04-vision-io-stb.md` — D1/D2/D3.
2. `libtorchmedia/include/torchmedia/_audio/_io.hpp` — the dr_wav ODR-gating pattern to mirror.

## Deliverables
- `_vision/_vendor/stb_image.h`, `stb_image_write.h` (vendored, license intact).
- `_vision/_io.hpp` (load_image/save_image); `vision.hpp` includes it.
- `unit_test/vision/io/main.cpp` round-trip test (define TORCHMEDIA_IO_IMPLEMENTATION).
- THIRD_PARTY_LICENSES updated.

## Steps
1. Vendor stb headers; gate impl behind TORCHMEDIA_IO_IMPLEMENTATION.
2. load_image: stbi_load -> from_blob [H,W,C] uint8 -> /255 float -> permute to [C,H,W]; stbi_image_free.
3. save_image: [C,H,W] -> clamp/round to [H,W,C] uint8 -> stbi_write_png.
4. Round-trip test (tolerance ~1/255); ctest green; 100% coverage of _vision/_io.hpp.

## Acceptance criteria
- [x] save -> load round-trip matches within ~1/255; shape [C,H,W] preserved.
- [x] `ctest` green (vision_test_io + others); 100% line/branch coverage of `_vision/_io.hpp`.
- [x] No new system/link dependency (stb is header-only, vendored).

## Constraints
- torch-native tensors; reuse the TORCHMEDIA_IO_IMPLEMENTATION macro (don't add a second impl macro for users).

## Notes / Assumptions
- Default save format is PNG (lossless); inferred from the .png extension.
