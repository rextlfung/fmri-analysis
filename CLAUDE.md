# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running tests

```julia
# From a Julia session at the project root:
using Pkg; Pkg.test()

# Or directly:
julia --project test/runtests.jl
```

## Loading the library interactively

Scripts are designed for interactive Julia sessions with hot-reloading:

```julia
using Revise
includet("src/fmri_analysis.jl")
using .FmriAnalysis
```

`Revise` is required before `includet` — without it, changes to the source file won't be picked up without restarting Julia.

## Architecture

### Module layout

`src/fmri_analysis.jl` defines the `FmriAnalysis` module and, at the end of the file, `include`s four files:
- `../scripts/run_analysis.jl` — high-level `analyze_and_plot` pipelines
- `../scripts/compare_recons.jl` — `compare_recons` multi-reconstruction spatial comparison driver
- `../scripts/compare_recons_time_series.jl` — `compare_recons_time_series` time-series comparison driver
- `src/export.jl` — NIfTI export helpers

There is no separate entry point — these five files together constitute the module. All included files rely on their dependencies (`Statistics`, `LinearAlgebra`, `FFTW`, `MAT`, `NIfTI`, `Printf`, `Plots`, `CairoMakie`, `Distributions`, `SpecialFunctions`, and the section 1–8 functions) being imported/defined by the parent module and must not add their own `using` statements. Note the unusual direction: a module under `src/` reaches up into `../scripts/` for the pipeline files.

Section 8 ("Shared Internal Helpers") defines internal functions used across the pipeline files to avoid duplication:
- `_load_recon(recon, scheme_base, n_discard)` — loads a reconstruction tuple (`:basic`/`:mslr`) into a 4-D Float32 array
- `_brain_glm(Y_4d, mask_flat, params; design_matrix)` — runs the reshape→mask→GLM pipeline on brain voxels, returns a named tuple `(t_brain, z_brain, beta, X, df, Y_brain)`
- `_unflatten_to_volume(scores, mask_flat, vol_dims; T)` — scatters brain-voxel scores into a full 3-D volume
- `_normalization_params(ref, mode)` — computes `(offset, scale)` for a normalization mode (`"demean"`/`"zscore"`/`"psc"`/`"none"`)
- `_write_nifti(path, data; success_msg)` — writes a NIfTI if the file doesn't already exist
- `_maybe_save_figure(fig, save_dir, filename)` — saves a CairoMakie figure if both arguments are non-nothing
- `STAT_COLORMAP` — the shared diverging colormap used for stat-map overlays

### GLM pipeline

The pipeline operates in this order: `build_design_matrix` → `fit_glm` → `compute_tscores` → `t_to_z`. The `run_glm` wrapper accepts an optional `design_matrix` keyword argument; pass a pre-built matrix when calling it repeatedly with the same parameters (e.g. across MSLR scales) to avoid redundant FFT convolutions. `run_glm` returns `(t_map, beta, X, z_map, df)`.

The GLM is fit on **brain voxels only**. The helper `_brain_glm` encapsulates the common workflow:
1. Flatten 4-D volume to `(n_scans × n_voxels)` matrix (masking before transpose for efficiency)
2. Fit GLM on the masked subset via `run_glm`
3. Return brain-space results as a named tuple; use `_unflatten_to_volume` to scatter back into full 3-D volumes

`analyze_and_plot` is a single function with two methods dispatched on input dimensionality (see `scripts/run_analysis.jl`):
- The **4-D method** `analyze_and_plot(X::…,4}, params, title_base; …)` returns four values: `(slice_idx, t_vol, Y_masked, z_vol)`.
- The **5-D method** `analyze_and_plot(X::…,5}, params, Nscales, patch_sizes, title_base; …)` returns seven values: `(slice_idx, t_vols, Y_vols, t_sum_vol, Y_sum_masked, z_vols, z_sum_vol)`.

Destructure at least the number of values you need — extra trailing values are silently ignored by Julia. Assigning to a single variable captures a tuple, which will fail when used as a `ref_slice_idx`.

### Brain masking

Brain masking shells out to FSL `bet` via `bet_brain_mask`. The function writes a temporary NIfTI to `/tmp`, calls `bet`, reads the mask back, then cleans up. There is no pure-Julia fallback.

Three environment variables must be set before running any analysis:
- `FSLDIR` — path to the FSL installation root (e.g. `/home/user/fsl`)
- `PATH` — must include `$FSLDIR/bin` so `bet` is found
- `FSLOUTPUTTYPE=NIFTI_GZ` — `bet_brain_mask` reads `*_mask.nii.gz`; other output types will cause a file-not-found error

To run a batch script from the shell: `FSLDIR=... FSLOUTPUTTYPE=NIFTI_GZ PATH="$FSLDIR/bin:$PATH" julia --project=.. <script>.jl`

### MSLR data format

Standard reconstructions are 4-D `(nx, ny, nz, nt)`. MSLR reconstructions are 5-D `(nx, ny, nz, nt, Nscales)`. The 5-D method of `analyze_and_plot` handles the 5-D case: it builds one shared brain mask from the temporal mean of the summed reconstruction, builds the design matrix once, then loops over scales.

### compare_recons pipeline (`scripts/compare_recons.jl`)

`compare_recons(schemes, recons, params; threshold_quantile=0.99f0, stat="t", slice_indices=nothing, save_dir=nothing, save_name=nothing)` loops over sampling schemes and produces one CairoMakie figure per scheme. Each figure has three rows (axial / coronal / sagittal) and one column per reconstruction. Slices are centred at the peak positive t-score voxel of the first recon (or at `slice_indices` if provided) and are shared across all columns. Pass `stat="z"` to display z-score maps instead of t-score maps. When `save_dir` and `save_name` are both provided, each figure is also saved as a PNG. Returns the slice indices NamedTuple used.

`recons` is a vector of tuples with the following shapes:
- `(:basic, base_dir, identifier, label)` — loads `base_dir/$(scheme_base)_$(identifier).mat`, key `"img"` (4-D)
- `(:mslr,  base_dir, cfg,        label)` — loads `base_dir/$(cfg)/$(scheme_base).mat`, key `"X"` (5-D); **sums all scales**
- `(:mslr,  base_dir, cfg,        label, n::Int)` — same file; **extracts the n-th scale** (1-based)

The brain mask and GLM design matrix are computed once from the first recon and shared across all recons within a scheme.

Column titles show `"<label>\n|<stat>| threshold = X.XX  max |<stat>| = X.XX"`. The colormap range is the global max across all recons; the display threshold is the `threshold_quantile`-percentile of the first recon's brain voxels. The visualization functions (`tmap_summary`, `plot_tmap_flat`, `plot_tmap_slices`) accept a `stat` keyword (default `"t-score"`) to customize labels for arbitrary statistical maps.

### compare_recons_time_series pipeline (`scripts/compare_recons_time_series.jl`)

`compare_recons_time_series(schemes, recons, params; ...)` is a companion to `compare_recons` that produces time-series plots instead of spatial maps. It accepts the same `schemes`, `recons`, and `params` arguments. For each scheme it produces three figures by default (four with `show_residuals=true`):

1. **Peak voxel** — BOLD time series at the highest-positive-stat voxel, one line per reconstruction.
2. **Top-n% average** — mean BOLD time series across the top `top_percent`% active voxels (positive stat scores).
3. **Power spectrum** — frequency-domain comparison of the top-n% averaged signals, with the task fundamental frequency marked (enabled by default via `show_spectrum=true`).
4. **Residuals** — residual (data minus model) time series for the top-n% average (disabled by default; enable with `show_residuals=true`).

Task blocks are shaded per-condition with distinct colors. A summary table is printed to the console showing peak stat value, R², voxel count, and residual standard deviation per reconstruction.

Note: the code computes fitted GLM model overlays, SEM bands, and per-recon stat annotations, but these are not currently wired into the plotting calls — the figures show only the raw time-series lines and task-block shading.

Key keyword arguments: `normalize` (`"demean"` / `"zscore"` / `"psc"` / `"none"`), `peak_source` (`:first` or `:per_recon`), `stat` (`"t"` or `"z"`), `top_percent` (default `1.0`), `condition_names`, `brain_mask`, `design_matrix`, `show_model` (accepted but unused), `show_task_shading`, `show_spectrum`, `show_residuals`, `time_range`, `save_dir`, `save_name`. Returns a named tuple `(peak_voxel_idx, top_voxel_indices)` with the brain-voxel indices used.

### Experiment scripts

Files in `experiments/` are named by session date (e.g. `20260409tap.jl`) and are structured for cell-by-cell execution in VS Code with the Julia extension (`# %%` cell markers). They are not importable modules. Each script sets `params` (an `ExperimentParams`) near the top; the first reconstruction analyzed in a session establishes a `ref_slice_idx` that is passed to all subsequent calls to keep plots at the same anatomical location.

`<session>_compare_recons.jl` companion scripts (e.g. `20260409tap_compare_recons.jl`) use `compare_recons` to produce side-by-side comparison figures. They define `schemes`, `recons`, and `params` then call `compare_recons(schemes, recons, params)`.

`<session>_compare_recons_time_series.jl` companion scripts use `compare_recons_time_series` for time-series comparison. They share the same `schemes`/`recons`/`params` definitions.
