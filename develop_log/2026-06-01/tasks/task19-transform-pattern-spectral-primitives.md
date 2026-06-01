# Task 19 — Transform pattern + window-buffer spectral primitives
id: 2026-06-01/task19
parent: 2026-06-01/progress02
status: done              # active | blocked | done
owner: code_agent

## Result
Done 2026-06-01. Established the `torchmedia::audio::transform` class idiom (ctor caches buffers,
`operator()`+`forward` alias, `xxx_option` fluent setters, default-derivation in ctor, a cached
`window()` accessor) in NEW `_transform_spectral.hpp` (aggregated by `_transform.hpp`, already
included via `audio.hpp`). Implemented `Spectrogram`, `InverseSpectrogram`, `GriffinLim`,
`SpectralCentroid`, each caching a Hann `window_` and delegating to the matching `functional::` op.
NEW ctest target `audio_test_transform` (`unit_test/audio/transform/{CMakeLists.txt, main.cpp,
gen_golden.py}`, wired into `unit_test/audio/CMakeLists.txt`). Tests use two checks per class:
delegation-equivalence to the golden-verified functional op (exact, no venv) + baked
torchaudio.transforms 2.5.1 golden (deterministic ramp input). `audio_test_transform` 29/29;
`ctest` 5/5 green. **100% line/region/function/branch coverage of `_transform_spectral.hpp`**
(223/223 lines, clang `-fprofile-instr-generate`/`llvm-cov`, `_vendor` excluded). Decisions held:
PascalCase classes + snake_case `xxx_option` (D2), independent transform option structs (D5),
`forward` alias provided (D1). No functional change needed (G1 deferred to task21).

## Objective
Establish the `torchmedia::audio::transform` **class idiom** (constructor caches buffers,
`operator()`/`forward` runs, `xxx_option` config with fluent setters, string→enum options, the
header split, and the `audio_test_transform` ctest target), and implement the four window-caching
spectral primitives: `Spectrogram`, `InverseSpectrogram`, `GriffinLim`, `SpectralCentroid`.

## Scope
In:
- The shared class pattern + header split (`_transform_spectral.hpp` and the `_transform.hpp`
  aggregator) per progress02 D1/D2/D5/D6.
- `transform::Spectrogram` — caches `window`; `operator()(waveform)` → `functional::spectrogram`.
- `transform::InverseSpectrogram` — caches `window`; `operator()(spec[, length])` →
  `functional::inverse_spectrogram`.
- `transform::GriffinLim` — caches `window`; `operator()(specgram)` → `functional::griffinlim`;
  momentum validation `0 <= momentum < 1`.
- `transform::SpectralCentroid` — caches `window`; `operator()(waveform)` →
  `functional::spectral_centroid`.
- The NEW test target `unit_test/audio/transform/` (`audio_test_transform`), wired into ctest.
Out:
- All other transforms (later tasks). `MelScale`/`MelSpectrogram` are task20 even though they may
  share `_transform_spectral.hpp` — land the file here, add to it there.
- Any functional change (none needed for this task).

## Inputs (read first, priority order)
1. `develop_log/2026-06-01/progress02-audio-transform-layer.md` — D1 (class idiom), D2 (PascalCase),
   D5 (independent option structs + enums), D6 (file split), D7 (testing).
2. `libtorchmedia/include/torchmedia/_audio/_functional.hpp` — signatures of `spectrogram`,
   `inverse_spectrogram`, `griffinlim`, `spectral_centroid`; the `xxx_option` fluent-setter style.
3. `libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp` — `spectrogram_option`,
   `griffinlim_option` field sets to mirror in the transform option structs.
4. `unit_test/audio/functional/{CMakeLists.txt, main.cpp, gen_golden.py}` and
   `unit_test/audio/CMakeLists.txt`, `unit_test/test_util.hpp` — the test scaffold to copy.
5. torchaudio v2.5.1 `transforms/_transforms.py` classes `Spectrogram`, `InverseSpectrogram`,
   `GriffinLim`, `SpectralCentroid` (default derivation, validation, what they pass to `F.*`).

Code to inspect/change:
- NEW `libtorchmedia/include/torchmedia/_audio/_transform_spectral.hpp`.
- `libtorchmedia/include/torchmedia/_audio/_transform.hpp` — turn the empty namespace into the
  aggregator (`#include` the themed headers; keep header guard).
- NEW `unit_test/audio/transform/{CMakeLists.txt, main.cpp, gen_golden.py}`.
- `unit_test/audio/CMakeLists.txt` — `add_subdirectory(transform)`.

## Deliverables
- `_transform_spectral.hpp` (guard `LIB_TORCH_MEDIA_AUDIO_TRANSFORM_SPECTRAL_HPP`,
  `#include "_functional.hpp"`, namespace `torchmedia::audio::transform`) defining the four classes,
  each with a sibling `xxx_option` struct (snake_case, fluent setters returning `*this`):
  - `class Spectrogram { explicit Spectrogram(spectrogram_option opt = {}); tensor_t
    operator()(const_tensor_lref_t waveform) const; tensor_t forward(const_tensor_lref_t w) const
    { return (*this)(w); } private: spectrogram_option opt_; tensor_t window_; };`
  - `InverseSpectrogram` (call op also accepts an optional `c10::optional<int64_t> length`).
  - `GriffinLim`, `SpectralCentroid` — same shape.
- `_transform.hpp` aggregates `_transform_spectral.hpp` (and reserves includes for later headers).
- `unit_test/audio/transform/` target `audio_test_transform` registered with ctest; `main.cpp`
  defines `TORCHMEDIA_IO_IMPLEMENTATION` like the other test mains if it loads audio (else not
  needed — these tests can build tensors directly).
- Golden cross-check tests for each of the four classes; golden block in `gen_golden.py`.

## Steps
1. **Class idiom (write once, reuse everywhere).** A transform = `{ ctor(option), private cached
   buffers, const operator()(tensor), forward alias }`. The constructor: (a) derives defaults
   (`win_length = win_length>0 ? win_length : n_fft`; `hop_length = hop_length>0 ? hop_length :
   win_length/2`), (b) validates (throw via `handle_exceptions<tensor_t,std::invalid_argument>`
   style used in functional, or a small `transform`-local check helper), (c) materializes cached
   tensors. Window materialization: if `opt_.window` is defined use it, else
   `torch::hann_window(opt_.win_length)` (or the `window_fn` the option selects — default hann).
2. **Header split + aggregator.** Create `_transform_spectral.hpp`; make `_transform.hpp`
   `#include` it. Confirm `audio.hpp` already includes `_transform.hpp` (it does) so
   `torchmedia.hpp` re-exports transitively. Keep include order: `_functional*.hpp` before
   `_transform*.hpp`.
3. **`Spectrogram`** — option mirrors `functional::spectrogram_option` fields (pad, n_fft,
   hop_length, win_length, power, normalized, center, pad_mode, onesided, return_complex,
   optional explicit `window`). `operator()` builds a `functional::spectrogram_option` from `opt_`
   with `.window(window_)` and returns `functional::spectrogram(waveform, that_option)`.
4. **`InverseSpectrogram`** — `operator()(spec, length=nullopt)` returns
   `functional::inverse_spectrogram(spec, length, pad, window_, n_fft, hop_length, win_length,
   normalized, center, pad_mode, onesided)`. Mind functional's `normalized` is a string
   ("none"/"window"/"frame_length") — map the bool/string option through faithfully.
5. **`GriffinLim`** — validate `0 <= momentum < 1` in the ctor; `operator()` builds a
   `functional::griffinlim_option` (with `window_`) and calls `functional::griffinlim`.
6. **`SpectralCentroid`** — `operator()` returns `functional::spectral_centroid(waveform,
   sample_rate, pad, window_, n_fft, hop_length, win_length)`.
7. **Test target + tests.** Copy the functional test scaffold to `unit_test/audio/transform/`
   (`CMakeLists.txt` builds `audio_test_transform`, links torch + fmt, registers `add_test`).
   For each class: a shape test, a "caching is correct" test (construct once, call twice, identical
   output), and a golden cross-check vs baked `torchaudio.transforms` values. Add
   `add_subdirectory(transform)` to `unit_test/audio/CMakeLists.txt`. Append a `gen_golden.py`
   block building fixed-seed inputs and printing `T.Spectrogram(...)(x)` etc.; bake the literals.
8. **Build, test, coverage.** `cmake --build build --target audio_test_transform && ctest
   --test-dir build` green; 100% line coverage of `_transform_spectral.hpp` for the four classes
   (every validation branch + the explicit-window vs default-window path).

## Acceptance criteria
- [ ] The four classes exist in `_transform_spectral.hpp`, `torchmedia::audio::transform`,
      header-only, with a cached `window_` member and `operator()`+`forward`.
- [ ] `_transform.hpp` aggregates the new header; project builds; `torchmedia.hpp` re-exports.
- [ ] `audio_test_transform` target exists, registered with ctest; `ctest --test-dir build` green
      (now 5 targets).
- [ ] Each class matches baked torchaudio 2.5.1 golden values (atol 1e-5, rtol 1e-4).
- [ ] Calling a constructed transform twice yields identical results (window cached, not rebuilt).
- [ ] `GriffinLim` rejects `momentum` outside `[0,1)`.
- [ ] 100% line coverage of `_transform_spectral.hpp` (vendored excluded).

## Constraints
- Header-only; classes with inline methods; torch-native ATen only. No new .cpp.
- PascalCase classes, snake_case `xxx_option`. Keep `clang-format` clean (LLVM, indent 4, col 120,
  namespaces indented).
- Do NOT reimplement spectrogram/griffinlim math — delegate to `functional::`.
- Cache the window in the ctor; never rebuild it in `operator()`.

## Notes / Assumptions
- Assumption: `operator()` is `const` for all four (no runtime mutation here — the first task to
  break this is `MVDR` in task25).
- Assumption: a `window_fn` selector can be a simple enum or just "explicit window tensor or
  default hann"; default-hann is enough to match torchaudio's default. Confirm with Mux only if a
  non-hann default is wanted.
- Dependency: this task UNBLOCKS task20 (composition needs `Spectrogram`). Keep the `Spectrogram`
  ctor/`operator()` signature stable.
- Question for Mux: confirm the `forward` alias is wanted (torchaudio parity) vs `operator()` only.
