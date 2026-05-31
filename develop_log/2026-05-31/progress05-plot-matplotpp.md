# Progress 05 — Optional plotting module (torchmedia::plot via matplot++)
id: 2026-05-31/progress05
date: 2026-05-31
author: human+ai
status: done
refs: [2026-05-31/progress04]
supersedes:
commits: []
files:
  - dependence/matplotplusplus (vendored, gitignored like other deps)
  - libtorchmedia/include/torchmedia/plot.hpp, _plot/_plot.hpp
  - CMakeLists.txt (TORCHMEDIA_WITH_PLOT option), libtorchmedia/CMakeLists.txt (torch_media_plot target)
  - unit_test/plot/{main.cpp, CMakeLists.txt}
  - script/download_dep (add matplot++)

## Goal
Add an OPTIONAL plotting module `torchmedia::plot` built on matplot++ (gnuplot backend) for waveform /
spectrogram / image visualization. Core audio/vision stay header-only; plot is opt-in.

## Context / Motivation
Maintainer wants C++-side plotting (was using a Python matplotlib script). After weighing self-render /
SVG / matplot++, chose matplot++ (the fullest non-Python option). Trade-off accepted: plot links
matplot++ and needs gnuplot at runtime (gnuplot 6.0 present locally).

## Decisions

### D1 — plot is an opt-in, non-header-only module
- CMake option `TORCHMEDIA_WITH_PLOT` (default OFF). When ON: `add_subdirectory(dependence/matplotplusplus)`
  and expose a `torch_media_plot` target linking `Matplot++::matplot`. `torchmedia.hpp` does NOT include
  `plot.hpp` (otherwise every consumer would be forced to have gnuplot).
- gnuplot is a runtime dependency; document it in CLAUDE.md / README.

### D2 — API
- `torchmedia::plot`, functions taking torch tensors:
  - `save_waveform(wav, path)`: [T] or [C,T] -> line plot.
  - `save_spectrogram(spec, path)`: [F,T] -> imagesc heatmap (+ colorbar).
  - `save_image(img, path)`: [C,H,W] -> image (or reuse vision::io for PNG).
- tensor -> std::vector<double> / vector<vector<double>> bridge.
- Optional `Plotter()` chaining wrapper as a follow-up.

### D3 — Tests are smoke-only
- gnuplot output is not byte-stable across versions, so the test verifies the call succeeds and the
  output file is created — no golden pixel compare.

## Tasks
- [x] [task08 — torchmedia::plot via matplot++](tasks/task08-plot-matplotpp.md)
- [x] [task09 — Plotter chaining API](tasks/task09-plotter-api.md)

## Issues / Gotchas
- matplot++ is a sizable CMake build; first configure/build is slow.
- plot is intentionally excluded from torchmedia.hpp; consumers include <torchmedia/plot.hpp> explicitly.
- gnuplot 6.0 + matplot++ prints many "Reading from '-' inside a multiplot ... use a datablock instead"
  warnings (matplot++ uses inline '-' data inside a multiplot for colorbar; gnuplot 6.0 deprecated that).
  The images render CORRECTLY (verified: waveform line + spectrogram heatmap + colorbar). It is upstream
  noise, not fatal. Follow-up: bump matplot++ or suppress the gnuplot stderr stream.

## Open / TODO (carry-over)
- colormap options; mel/MFCC convenience plots; gnuplot 6.0 warning suppression.
- Earlier carry-overs: vision Tier-2 transforms, mel_filter_bank vectorization.

## Agent log
- 2026-05-31 [ai] gnuplot 6.0 present, github reachable. Cloning matplot++ into dependence/. Authored
  progress05 + task08. Integrating once the clone completes.
- 2026-05-31 [ai] Done: cloned matplot++ (dependence/, gitignored); added TORCHMEDIA_WITH_PLOT option +
  torch_media_plot target; implemented torchmedia::plot (save_waveform line, save_spectrogram imagesc+colorbar)
  over torch tensors; smoke test + plot_test ctest target. Built with -DTORCHMEDIA_WITH_PLOT=ON; both PNGs
  render correctly (verified visually). Core build unchanged with the option OFF. matplot++ added to
  download_dep; gnuplot noted in CLAUDE.md. (gnuplot 6.0 multiplot warning recorded above — images are correct.)
- 2026-05-31 [ai] task09: added Plotter() chaining API (heatmap/waveform/title/xlabel/ylabel/colorbar/save);
  refactored save_waveform/save_spectrogram onto it. plot_test 4/4 green; fixed an async-gnuplot test flake
  by polling for output files instead of checking exists() immediately after save.
