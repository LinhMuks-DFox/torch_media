# Task 17 — apply_codec: decide scope, then (optionally) the WAV-lossless requantization subset
id: 2026-06-01/task17
parent: 2026-06-01/progress01
status: blocked             # active | blocked | done (Mux decision 2026-06-01)
owner: code_agent

## Objective
Capture Mux's scope decision for `apply_codec`, then (only if approved) implement the single
header-only-feasible slice — a lossless WAV requantization round-trip in
`libtorchmedia/include/torchmedia/_audio/_functional.hpp` (reusing the vendored `dr_wav` save/load in
`_io.hpp`) — leaving all compressed codecs behind the optional `TORCHMEDIA_WITH_FFMPEG` path.

## Scope
In:
- **Decision first (blocking):** record Mux's choice — (a) DEFER/SKIP entirely, (b) ship only the
  WAV-lossless requantization subset, or (c) wire a real codec round-trip under
  `TORCHMEDIA_WITH_FFMPEG`. No code lands before this is answered in the progress `Agent log`.
- **If (b) approved only:** `apply_codec(...)` restricted to `format == "wav"` (and `format == "pcm"`),
  simulating the `encoding` / `bits_per_sample` quantization via the vendored `dr_wav`
  write-then-read (save_audio → load_audio) on a `std::filesystem` temp path. This is the *only*
  tensor-observable behavior of `apply_codec` reproducible header-only: lossy bit-depth
  requantization (e.g. 16→8-bit PCM) plus the sox→torch resample-back when sample rates differ.
- `apply_codec_option` (if needed to carry `format`/`compression`/`encoding`/`bits_per_sample`),
  following the `xxx_option` fluent-setter convention.
Out:
- Autograd / backward (forward-only; a codec round-trip is non-differentiable by nature).
- **All compressed codecs** (mp3 / ogg-vorbis / flac / gsm / amr / opus) — these have NO torch-native
  equivalent; their lossy behavior is the codec's, not reproducible with ATen ops. Gate behind
  `TORCHMEDIA_WITH_FFMPEG` (default OFF, per `2026-05-30/progress01`) — design only, no implementation
  in this task unless Mux picks option (c).
- The deprecated upstream behaviors beyond what `dr_wav` exposes (e.g. sox-specific `compression`
  semantics for lossless formats).
- Transform-layer wrapper — none exists upstream for `apply_codec`.

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress01-functional-full-port.md` — parent; D4 gap entry lines 82–85
   (`apply_codec` XL, **recommend DEFER/SKIP**, only a lossless WAV requantization subset is feasible
   via vendored `dr_wav`; full support belongs behind `TORCHMEDIA_WITH_FFMPEG`); D6 line 98 lists
   task17 under **Defer/decide**.
2. `develop_log/2026-05-30/progress01-remove-ffmpeg-dr_wav-io.md` — the FFmpeg/dr_wav I/O decision:
   FFmpeg is optional (`TORCHMEDIA_WITH_FFMPEG`, default OFF) for compressed encode only; WAV via
   `dr_wav`; `load_audio` returns the native sample rate, resampling is a separate torch-native op.
3. torchaudio v2.5.1 source (authoritative signature / body / deprecation):
   `https://raw.githubusercontent.com/pytorch/audio/v2.5.1/src/torchaudio/functional/functional.py`
   — `apply_codec(waveform, sample_rate, format, channels_first=True, compression=None,
   encoding=None, bits_per_sample=None)`. Body: open a `NamedTemporaryFile`, `save()` via the sox
   backend (encoding the chosen `format`/`compression`/`encoding`/`bits_per_sample`), `load()` it
   back, and if the loaded sample rate differs, `resample(augmented, loaded_sr, sample_rate)`.
   Decorated `@deprecated(... torchaudio.io.AudioEffector ..., remove=False)`.

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — where the new `apply_codec` would live;
  model error handling on `handle_exceptions<T, ExceptionT>(...)` / `TORCH_CHECK` already used there;
  reuse the ported `resample` (the resample-back leg).
- `libtorchmedia/include/torchmedia/_audio/_io.hpp` — `save_audio` (lines 62–94) currently **hardcodes
  `format.bitsPerSample = 16` / `DR_WAVE_FORMAT_PCM`** and `drwav_f32_to_s16`; `load_audio` (line 34)
  returns native sample rate. The WAV-lossless subset needs `save_audio` parameterized by
  `bits_per_sample` (8/16/24/32 PCM, optionally f32 IEEE) to make requantization observable.
- `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` — add `apply_codec_option`
  only if Mux picks (b)/(c) and the param set warrants it.
- `unit_test/audio/functional/main.cpp` — add the smoke test + register in `main()`.
- `unit_test/audio/functional/gen_golden.py` — only if a torchaudio cross-check is wanted (see Notes:
  a deterministic WAV-lossless round-trip can be self-referenced without `.venv`).

## Deliverables
- **Always:** a decision recorded in `develop_log/2026-06-01/progress01-functional-full-port.md`'s
  `Agent log` (DEFER / WAV-lossless subset / FFmpeg-gated), and this task's `status` set accordingly
  (`done` if DEFER is chosen with no code; `active`→implement otherwise).
- **If (b):** `inline auto apply_codec(tensor_t waveform, int sample_rate, const std::string &format,
  bool channels_first = true, c10::optional<double> compression = c10::nullopt,
  c10::optional<std::string> encoding = c10::nullopt,
  c10::optional<int> bits_per_sample = c10::nullopt) -> torch::Tensor;` in `_functional.hpp`,
  restricted to `format ∈ {"wav","pcm"}` (else raise — see Steps). A parameterized
  `save_audio(..., int bits_per_sample, ...)` overload in `_io.hpp` (default 16 preserves current
  behavior). Optional `apply_codec_option` in the options header.
- **If (c):** a compile-time `#ifdef TORCHMEDIA_WITH_FFMPEG` branch design (encode→decode→resample via
  the FFmpeg backend) — documented, not implemented in this task.
- Smoke test `test_apply_codec_wav_roundtrip` in `main.cpp` (registered in `main()`): only built/run
  under the WAV-lossless subset; compressed codecs are explicitly **not** tested header-only.

## Steps
1. **Decision (blocking).** Present Mux the three options (DEFER/SKIP, WAV-lossless subset,
   FFmpeg-gated) with the trade-offs from progress01 D4 and the upstream deprecation note. Write the
   answer into the progress `Agent log`. **Stop here and mark `status: done` if DEFER is chosen — no
   code, no test.** Otherwise continue.
2. **Validation / raises (subset path).** In `apply_codec`, require a 2-D `waveform`
   (`TORCH_CHECK(waveform.dim() == 2, ...)`); if `!channels_first`, transpose to `[channels, frames]`
   for the save and transpose the result back. Raise `std::invalid_argument` via
   `handle_exceptions<torch::Tensor, std::invalid_argument>(...)` for any `format` other than
   `"wav"`/`"pcm"` with a message naming `TORCHMEDIA_WITH_FFMPEG` as the path for compressed formats.
   Reject `bits_per_sample` not in `{8,16,24,32}` (PCM) — the only depths `dr_wav` can write losslessly.
3. **Parameterize the writer.** Extend `_io.hpp` `save_audio` with a `bits_per_sample` parameter
   (overload, default 16 to keep existing call sites): set `format.bitsPerSample` accordingly and use
   the matching `drwav_f32_to_s16` / `drwav_f32_to_s24` / `drwav_f32_to_s32` (or `DR_WAVE_FORMAT_PCM`
   widths). This is what makes 16→8-bit requantization *observable*; without it the round-trip is a
   no-op and the "codec" has no effect to simulate.
4. **Round-trip body.** Build a unique temp path under `std::filesystem::temp_directory_path()` (e.g.
   `apply_codec_<pid>_<counter>.wav`); `save_audio(work, tmp, sample_rate, bits_per_sample)`;
   `auto loaded = load_audio(tmp);` then `std::filesystem::remove(tmp)`. If
   `loaded.sample_rate != sample_rate`, `augmented = resample(loaded.waveform, loaded.sample_rate,
   sample_rate)` (the resample-back leg, mirroring upstream); else `augmented = loaded.waveform`.
   Restore the original layout if `!channels_first`. Return `augmented`. (For WAV at the same rate the
   only lossy effect is the bit-depth quantization from step 3.)
5. **Add tests, ctest green, coverage.** In `main.cpp`, add `test_apply_codec_wav_roundtrip`:
   16-bit round-trip of a small fixed `[1, N]` float waveform must satisfy
   `TM_CHECK_TENSOR_CLOSE(out, input, /*atol*/ 1.0/32768.0, /*rtol*/ 0)` (1-LSB s16 quantization);
   add an 8-bit case asserting the output is *coarser* (e.g. distinct-value count drops / error grows
   to ~1/256), proving the requantization is observed; add a raise case for an unsupported `format`
   (e.g. `"mp3"`). Register in `main()`. This is a **self-referenced** golden (closed-form
   quantization step) — no `.venv` needed; only extend `gen_golden.py` if a torchaudio WAV cross-check
   is explicitly requested by Mux. Build & run:
   `cmake --build build --target audio_test_functional &&
   ./build/unit_test/audio/functional/audio_test_functional`; `ctest --test-dir build` green; 100%
   line coverage of every new line in `apply_codec` and the new `save_audio` overload (each `format`
   branch, each `bits_per_sample` width exercised by a test, the resample-back branch reached or
   documented unreachable for same-rate WAV).

## Acceptance criteria
- [ ] Mux's scope decision is recorded in `progress01`'s `Agent log`; this file's `status` reflects it.
- [ ] **(DEFER chosen):** no code added; task marked `done`; progress task-list line for task17 notes
      the DEFER rationale + the `AudioEffector`/`TORCHMEDIA_WITH_FFMPEG` pointer. (Remaining criteria N/A.)
- [ ] **(WAV-lossless subset chosen):** 16-bit round-trip matches the input within 1-LSB
      (`atol = 1/32768`); the 8-bit case demonstrably coarsens the signal; an unsupported `format`
      raises `std::invalid_argument` naming `TORCHMEDIA_WITH_FFMPEG`.
- [ ] `save_audio`'s existing 16-bit call sites are unchanged (parameter defaulted), no I/O test regresses.
- [ ] `ctest --test-dir build` green; 100% line coverage of the new `_functional.hpp` / `_io.hpp` lines
      (vendored `_vendor/dr_wav.h` excluded).

## Constraints
- Header-only: `inline` free functions in `torchmedia::audio::functional`; torch-native ops only for
  the resample-back leg (delegates to the ported `resample`). The codec round-trip itself uses the
  **vendored** `dr_wav` via `_io.hpp` — no new heavy/system dependency.
- **No tensor math can reproduce a real codec** — this is a scope-pending task; the only header-only
  effect is lossless WAV bit-depth requantization. Compressed formats MUST be refused (or gated behind
  `TORCHMEDIA_WITH_FFMPEG`, default OFF), never silently no-op'd.
- Match torchaudio's surface (signature/param names/`channels_first` handling, resample-back when the
  loaded rate differs) for the WAV path; do not claim parity for compressed formats.
- Touching `_io.hpp` `save_audio` must stay backward-compatible (default `bits_per_sample = 16`); the
  vendored implementation stays gated behind `TORCHMEDIA_IO_IMPLEMENTATION` / `DR_WAV_IMPLEMENTATION`
  (do not move the impl macro).
- Temp files: unique per call, removed in all paths (including on exception) — no leaks under repeated
  test runs.

## Notes / Assumptions
- Assumption: `resample` (task05 of `2026-05-31`, the merged torch-native resample) is available for
  the resample-back leg; for the WAV same-rate subset that leg is typically not exercised, so the test
  should either force a rate change or the branch be documented unreachable.
- Assumption: `_io.hpp`'s `save_audio` currently writes **only 16-bit PCM** (`bitsPerSample = 16`,
  `drwav_f32_to_s16`) — confirmed lines 75–83; this is why a `bits_per_sample` parameter is required to
  make requantization observable. `load_audio` returns the native sample rate (D from 2026-05-30).
- Gotcha: upstream `apply_codec` is **deprecated** (superseded by `torchaudio.io.AudioEffector`,
  `remove=False`); porting even the WAV subset is low priority — the DEFER recommendation stands unless
  Mux wants the augmentation primitive for parity.
- Gotcha: there is **no closed-form tensor truth** for any compressed codec; do not attempt golden
  values for mp3/ogg/flac — they are untestable without the actual codec, hence FFmpeg-gated only.
- Gotcha: `dr_wav` write supports PCM widths 8/16/24/32 and IEEE f32; "lossless" only holds for the
  width chosen — a 16→8-bit round-trip is intentionally lossy and that loss IS the simulated codec.
- **Question for Mux:** Defer entirely (recommended), ship the WAV-lossless requantization subset, or
  invest in a `TORCHMEDIA_WITH_FFMPEG`-gated real round-trip? Answer this before any code is written.
