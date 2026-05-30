# Task 01 — Vendor dr_wav and rewrite the WAV I/O path (drop FFmpeg link)
id: 2026-05-30/task01
parent: 2026-05-30/progress01
status: done
owner: code_agent

## Objective
Replace the FFmpeg-based WAV load/save in `_io.hpp` with vendored header-only `dr_wav`, so the audio
I/O unit test builds and runs with zero FFmpeg linkage.

## Scope
In:
- Vendor `dr_wav.h` into `libtorchmedia/include/torchmedia/_audio/_vendor/`.
- Rewrite `load_audio` / `save_audio` WAV path on top of dr_wav.
- Gate the dr_wav implementation behind `TORCHMEDIA_IO_IMPLEMENTATION`.
- Remove FFmpeg from the build and `script/download_dep`.
- (folded in) Migrate the build from xmake to CMake, C++20 — see progress01 D6/D7.
Out:
- FLAC / MP3 / OGG decoders (later tasks).
- torch-native resample op (separate progress).
- The optional `TORCHMEDIA_WITH_FFMPEG` path.

## Inputs (read first, priority order)
1. `develop_log/2026-05-30/progress01-remove-ffmpeg-dr_wav-io.md` — decisions D1-D7 and the why.

Code touched:
- `libtorchmedia/include/torchmedia/_audio/_io.hpp` — FFmpeg load/save replaced by dr_wav.
- `unit_test/audio/io/main.cpp` — self-contained round-trip test.
- `CMakeLists.txt` + per-dir CMakeLists, `script/download_dep` — build, FFmpeg removed.

## Deliverables
- `libtorchmedia/include/torchmedia/_audio/_vendor/dr_wav.h` — vendored, license header intact. [done]
- `libtorchmedia/include/torchmedia/_audio/_io.hpp` — dr_wav load/save; `load_audio_t` + signatures
  unchanged; `[channels, samples]` f32; native sample rate; no per-channel clone. [done]
- CMake build (root + libtorchmedia + unit_test/audio/io); xmake files removed. [done]
- `THIRD_PARTY_LICENSES` — attribution for mackron/dr_libs. [done]

## Steps
1. **Vendor dr_wav** — added `_vendor/dr_wav.h` (v0.14.6), license header kept. [done]
2. **Implementation gating** — `#ifdef TORCHMEDIA_IO_IMPLEMENTATION` -> `#define DR_WAV_IMPLEMENTATION`
   then include the vendored header; declarations only elsewhere. [done]
3. **Rewrite load** — `drwav_open_file_and_read_pcm_frames_f32` -> `from_blob({frames, channels})`
   -> `.transpose(0,1).contiguous()`; `drwav_free`. [done]
4. **Rewrite save** — `[channels, samples]` -> CPU contiguous interleaved -> `drwav_f32_to_s16`
   -> `drwav_init_file_write` / `drwav_write_pcm_frames` (PCM16). [done]
5. **De-FFmpeg + CMake** — removed avformat/avcodec/avutil/swresample and all xmake `.lua`; added a
   CMake build (C++20) using `find_package(Torch)` over `dependence/libtorch`, fmt header-only. [done]
6. **Build & run** — io round-trip builds & runs with no FFmpeg linkage. [done]

## Acceptance criteria
- [x] `cmake --build build --target audio_test_io` builds and runs with no FFmpeg in the link line.
- [x] `load_audio` returns a `[channels, samples]` float32 tensor at the file's native sample rate.
- [x] save -> load round-trip on a PCM16 WAV matches within tolerance (max abs err 6.1e-05).
- [x] `load_audio_t` and the `load_audio` / `save_audio` signatures are unchanged (public API intact).
- [x] No `av*` / `swr*` symbols remain referenced in the default audio path.

## Constraints
- Keep the public I/O API unchanged. [held]
- No new system/link dependencies; dr_wav is header-only and vendored. [held]
- Do not add resampling in this task. [held]

## Result
Done 2026-05-30. dr_wav backend integrated; FFmpeg + xmake fully removed; CMake (C++20) build green for
all targets (`audio_test_io`, `audio_test_functional`, `vision_test_io`). Two bugs found & fixed during
verification: `load_audio("literal")` overload ambiguity (added `const char*` overloads), and a
`.gitignore` rule that was swallowing `CMakeLists.txt`.

## Notes / Assumptions
- `save_audio` defaults to PCM16 (compatibility). A float32 output option can come later.
