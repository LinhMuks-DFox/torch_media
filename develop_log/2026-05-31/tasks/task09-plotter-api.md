# Task 09 — Plotter chaining API
id: 2026-05-31/task09
parent: 2026-05-31/progress05
status: done
owner: code_agent

## Objective
Add a chaining `Plotter()` builder to `torchmedia::plot`; refactor `save_waveform`/`save_spectrogram`
to reuse it.

## Scope
In: `Plotter` class (waveform / heatmap / title / xlabel / ylabel / colorbar / save, all chainable);
convenience wrappers reuse it; plot smoke test exercises the chaining API.
Out: more chart types / colormap options (later).

## Inputs
1. `develop_log/2026-05-31/progress05-plot-matplotpp.md` — D2 (API) + Open/TODO (Plotter).

## Deliverables
- `Plotter` in `_plot/_plot.hpp`; `save_waveform`/`save_spectrogram` built on it.
- plot smoke test uses Plotter chaining; polls for gnuplot's async output.

## Acceptance criteria
- [x] `Plotter().heatmap(...).colorbar().title(...).save(...)` and `.waveform(...)` produce valid PNGs.
- [x] `plot_test` green (4/4), output files created.

## Result
Done 2026-05-31. `Plotter()` chaining implemented; convenience wrappers refactored onto it. Fixed a
smoke-test flake: gnuplot writes asynchronously, so the test now issues all saves then polls for the
output files (the files were always produced; the `exists()` check was just too eager).

## Notes / Assumptions
- Plotter uses matplot++'s global current-axes (cla per instance) — fine for single-instance chaining.
