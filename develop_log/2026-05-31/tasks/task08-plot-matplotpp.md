# Task 08 — torchmedia::plot via matplot++
id: 2026-05-31/task08
parent: 2026-05-31/progress05
status: done
owner: code_agent

## Objective
Add an opt-in `torchmedia::plot` module (matplot++ / gnuplot backend) for waveform / spectrogram /
image plotting, without affecting the header-only core.

## Scope
In:
- `_plot/_plot.hpp` + `plot.hpp`: `save_waveform`, `save_spectrogram`, `save_image` (torch tensors).
- CMake: `TORCHMEDIA_WITH_PLOT` option; `add_subdirectory(dependence/matplotplusplus)`; `torch_media_plot`
  target linking Matplot++::matplot.
- `unit_test/plot/` smoke test (output file is created); add matplot++ to `script/download_dep`.
Out:
- Plotter() chaining API; colormaps; golden pixel comparison.

## Inputs
1. `develop_log/2026-05-31/progress05-plot-matplotpp.md` — D1/D2/D3.

## Deliverables
- `_plot/_plot.hpp`, `plot.hpp` (NOT included by torchmedia.hpp).
- CMake option + torch_media_plot target; unit_test/plot smoke test registered with ctest (guarded by the option).
- download_dep fetches matplot++; CLAUDE.md notes the gnuplot runtime dep.

## Steps
1. CMake: TORCHMEDIA_WITH_PLOT -> add_subdirectory(matplot++) + torch_media_plot INTERFACE (link matplot).
2. Implement plot wrappers (tensor -> matplot data; matplot::plot / imagesc / save).
3. Smoke test: save a waveform + spectrogram PNG; assert files exist.
4. `cmake -DTORCHMEDIA_WITH_PLOT=ON` builds; ctest (plot test) green.

## Acceptance criteria
- [x] With TORCHMEDIA_WITH_PLOT=ON, the project builds and the plot smoke test produces output files.
- [x] With the option OFF (default), the core build is unchanged and stays header-only (no matplot++/gnuplot needed).

## Constraints
- Core audio/vision remain header-only; plot is fully opt-in.
- gnuplot is a documented runtime dependency.

## Notes / Assumptions
- Output format inferred from extension (PNG via gnuplot).
