# fMRI Analysis — Task-Based Activation Mapping (t-scores & z-scores)

Julia pipeline for task-based fMRI GLM analysis, designed to compare activation maps across multiple MRI reconstruction methods (SMS-EPI/slice-GRAPPA, CG-SENSE, L+S, LLR, and multi-scale low-rank / MSLR).

---

## Repository structure

```
src/
  fmri_analysis.jl       # Module FmriAnalysis: HRF, design matrix, GLM, correction, plotting
  export.jl              # NIfTI export helpers (included inside FmriAnalysis)
scripts/
  run_analysis.jl        # analyze_and_plot pipelines (included inside FmriAnalysis)
  compare_recons.jl      # compare_recons driver (included inside FmriAnalysis)
  compare_recons_time_series.jl  # compare_recons_time_series driver (included inside FmriAnalysis)
experiments/
  <session>.jl           # Per-session analysis scripts (one file per scan date)
  <session>_compare_recons.jl              # Side-by-side spatial comparison figures
  <session>_compare_recons_time_series.jl  # Side-by-side time-series comparison figures
test/
  runtests.jl            # Synthetic-data unit tests
```

---

## Core library (`src/fmri_analysis.jl`)

The file defines the `FmriAnalysis` module. Load it with:

```julia
includet("src/fmri_analysis.jl")
using .FmriAnalysis
```

### 1. Haemodynamic Response Function
- `canonical_hrf(tr)` — SPM-style double-gamma HRF sampled at `tr` seconds. Uses `SpecialFunctions.gamma`.

### 2. Design Matrix
- `build_design_matrix(onsets, durations, n_scans, tr)` — Constructs an `(n_scans × n_conditions + 1)` GLM design matrix via FFT-based convolution with 16× temporal oversampling. The HRF is unit-area normalized so a sustained stimulus yields a regressor amplitude of ≈1; this is a global per-column scale and does not affect t-maps. Last column is the intercept.

### 3. GLM Fitting, Contrast t-scores & z-scores
- `fit_glm(X, Y)` — OLS fit returning `β`, residuals, and `(X'X)⁻¹`.
- `compute_tscores(β, residuals, XtXinv, contrast)` — Voxel-wise t-scores for a contrast vector.
- `t_to_z(t, df)` — Convert t-scores to z-scores via the probability integral transform (FSL-style `sign(t) · Φ⁻¹(F_t(|t|; df))`). Uses log-space computation for numerical stability at extreme values.
- `run_glm(Y, onsets, durations, contrast, n_scans, tr; design_matrix=nothing)` — Full pipeline wrapper: design matrix → fit → t-scores → z-scores. Pass a pre-built `design_matrix` to avoid recomputing it across repeated calls. Returns `(t_map, beta, X, z_map, df)`.

### 4. Multiple Comparisons Correction
- `t_to_p(t, df; two_tailed=true)` — Convert t-scores to p-values via `Distributions.TDist`.
- `z_to_p(z; two_tailed=true)` — Convert z-scores to p-values via the standard normal distribution.
- `fdr_correct(t_map, df; q=0.05)` — Benjamini-Hochberg FDR correction; returns thresholded t-map, binary mask, raw p-values, and t-threshold.
- `bonferroni_correct(t_map, df; alpha=0.05)` — Bonferroni FWER correction; same return signature.

### 5. Brain Mask Extraction
- `bet_brain_mask(mean_vol; tmp_dir="/tmp")` — Runs FSL `bet` on a 3-D temporal mean volume and returns a `BitArray{3}` brain mask. Writes and cleans up temporary NIfTI files in `tmp_dir`. Requires FSL on `PATH`.

### 6. Visualisation
- `plot_tmap_flat(t_map)` — Two-panel Plots.jl figure: per-voxel bar chart + t-score histogram.
- `plot_design_matrix(X)` — Heatmap of the design matrix (sanity check).
- `plot_tmap_slices(t_vol)` — Orthogonal axial/coronal/sagittal slice view (CairoMakie) with optional anatomical underlay.
- `tmap_summary(t_map)` — Prints a console table of voxel counts surviving a range of |t| thresholds (with approximate p-values), plus summary statistics (mean, std, min/max, percentiles).

### 7. Experiment Parameters
- `ExperimentParams` — Struct holding `tr`, `onsets`, `durations`, `contrast`, and `n_discard` (leading frames to drop before fitting).

### 8. Analysis Pipelines (`scripts/run_analysis.jl`)
`analyze_and_plot` is a single function with two methods dispatched on input dimensionality:
- `analyze_and_plot(X, params, title; ref_slice_idx=nothing, brain_mask=nothing, design_matrix=nothing, tmp_dir="/tmp")` (4-D `X`) — Runs the full GLM pipeline on a single 4-D volume: derives a brain mask via `bet_brain_mask` (or reuses a passed-in one), fits the GLM on brain voxels only, prints a `tmap_summary`, and displays orthogonal slice plots thresholded at the 99th percentile of |t|. Returns `(slice_idx, t_vol, Y_masked, z_vol)`.
- `analyze_and_plot(X, params, Nscales, patch_sizes, title; ref_slice_idx=nothing, brain_mask=nothing, tmp_dir="/tmp", threshold_quantile=0.99, plot_summary=false, plot_sum=false)` (5-D `X`) — Runs the same pipeline on each scale of a 5-D MSLR reconstruction. The design matrix is built once and reused across all scales. Brain mask is derived from the temporal mean of the summed reconstruction (or reused if passed in). Returns `(slice_idx, t_vols, Y_vols, t_sum_vol, Y_sum_masked, z_vols, z_sum_vol)`.

### 9. Reconstruction Comparison (`scripts/compare_recons.jl`)
- `compare_recons(schemes, recons, params; threshold_quantile=0.99f0, stat="t", slice_indices=nothing, save_dir=nothing, save_name=nothing)` — For each sampling scheme, runs the GLM on every reconstruction and displays a single CairoMakie figure. The figure has one column per reconstruction and three rows (axial / coronal / sagittal) centred at the peak positive stat-score voxel of the first recon (or at `slice_indices` if provided); all columns share the same slice indices. Pass `stat="z"` to display z-score maps instead of t-score maps. Column titles show the threshold and max |stat| for that recon. The brain mask and design matrix are computed once per scheme (from the first recon) and shared. When `save_dir` and `save_name` are both provided, each figure is also saved as a PNG. Returns the slice indices NamedTuple used.

  `recons` entries are tuples of one of three shapes:
  - `(:basic, base_dir, identifier, label)` — 4-D reconstruction from `base_dir/$(scheme_base)_$(identifier).mat`
  - `(:mslr,  base_dir, cfg, label)` — 5-D MSLR reconstruction; sums all scales
  - `(:mslr,  base_dir, cfg, label, n::Int)` — same file; extracts the n-th scale (1-based)

### 10. Time-Series Comparison (`scripts/compare_recons_time_series.jl`)
- `compare_recons_time_series(schemes, recons, params; ...)` — Companion to `compare_recons` that produces time-series plots instead of spatial maps. Accepts the same `schemes`, `recons`, and `params` arguments. For each scheme it produces three figures by default (four with `show_residuals=true`):
  1. **Peak voxel** — BOLD time series at the highest-positive-stat voxel, one line per reconstruction.
  2. **Top-n% average** — mean BOLD time series across the top `top_percent`% active voxels (positive stat scores).
  3. **Power spectrum** — frequency-domain comparison of the top-n% averaged signals, with the task fundamental frequency marked.
  4. **Residuals** (opt-in) — residual (data minus model) time series for the top-n% average.

  Task blocks are shaded per-condition with distinct colors. A summary table is printed to the console showing peak stat value, R², voxel count, and residual standard deviation per reconstruction.

  Key keyword arguments: `normalize` (`"demean"` / `"zscore"` / `"psc"` / `"none"`), `peak_source` (`:first` or `:per_recon`), `stat` (`"t"` or `"z"`), `top_percent` (default `1.0`), `condition_names`, `brain_mask`, `design_matrix`, `show_task_shading`, `show_spectrum`, `show_residuals`, `time_range`, `save_dir`, `save_name`. Returns a named tuple `(peak_voxel_idx, top_voxel_indices)`.

### 11. NIfTI Export (`src/export.jl`)
- `export_niftis(Y_masked, t_vol, prefix, out_dir; z_vol=nothing)` — Exports a post-discard magnitude timeseries, t-score volume, and optionally z-score volume as NIfTI files for a single 4-D reconstruction.
- `export_niftis(Y_vols, t_vols, patch_sizes, Nscales, prefix, out_dir; z_vols=nothing)` — Same for a 5-D MSLR reconstruction; writes one magnitude + t-map (+ optional z-map) per scale.

---

## Experiment design (tapping paradigm)

- **Task:** finger-tapping, alternating tap/rest blocks
- **Block duration:** 20 s tap / 20 s rest (40 s period)
- **TR:** 0.8 s
- **Contrast:** tap > rest (`[1, -1, 0]`)
- **Instructional frames:** discarded before fitting (session-dependent; set via `n_discard` in `ExperimentParams`)

---

## Session scripts (`experiments/`)

Each `<session>.jl` script loads reconstructed volumes from disk, defines an `ExperimentParams`, and calls `analyze_and_plot` directly. The first reconstruction analyzed establishes a reference slice index that is reused across all subsequent comparisons so that all plots show the same anatomical location.

Each `<session>_compare_recons.jl` companion script uses `compare_recons` to produce side-by-side spatial comparison figures across reconstruction methods. It defines `schemes` (3-tuples `(file_base, label, _)`) and `recons` (4- or 5-tuples as described in section 9) then calls `compare_recons(schemes, recons, params)`.

`<session>_compare_recons_time_series.jl` companion scripts use `compare_recons_time_series` for time-series comparison. They share the same `schemes`/`recons`/`params` definitions.

Reconstructions compared per session may include:

| Label | Description |
|---|---|
| SMS-EPI + slice-GRAPPA | Product reconstruction (NIfTI) |
| Gaussian / CAIPI / PD + CG-SENSE | Compressed-sensing or parallel-imaging recon (NIfTI) |
| L+S | Low-rank + sparse recon (MAT, scale 2 extracted) |
| LLR | Locally low-rank recon (MAT, scales summed) |
| MSLR (*N* scales) | Multi-scale low-rank recon (MAT, per-scale + summed) |

---

## Dependencies

### Julia packages

Add via the Julia package manager (`]add <pkg>`):

```julia
CairoMakie       # 3-D orthogonal slice visualisation
Plots            # flat t-map and design matrix plots
FFTW             # FFT-based HRF convolution
Distributions    # t-distribution CDF for p-value conversion
SpecialFunctions # gamma function for HRF construction
MAT              # loading .mat reconstruction files
NIfTI            # loading/writing .nii / .nii.gz volumes
Revise           # hot-reloading during interactive development
```

Standard library modules used (no installation needed): `Statistics`, `LinearAlgebra`, `Printf`, `Random`.

### System dependency

**FSL** must be installed. Used by `bet_brain_mask` to derive binary brain masks. Three environment variables are required before running any analysis:

```bash
export FSLDIR=/path/to/fsl          # e.g. /home/user/fsl or /usr/local/fsl
export PATH="$FSLDIR/bin:$PATH"     # puts bet and other FSL tools on PATH
export FSLOUTPUTTYPE=NIFTI_GZ       # bet must write .nii.gz (code expects this extension)
```

On most FSL installations you can source the provided setup script instead:

```bash
source "$FSLDIR/etc/fslconf/fsl.sh"
export PATH="$FSLDIR/bin:$PATH"
```

---

## Usage

```julia
# In a Julia session or notebook
using Revise
includet("src/fmri_analysis.jl")   # hot-reload on edits
using .FmriAnalysis

# Define experiment parameters
params = ExperimentParams(
    tr        = 0.8f0,
    onsets    = [collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
    durations = [fill(20.0f0, 9), fill(20.0f0, 9)],
    contrast  = [1.0f0, -1.0f0, 0.0f0],
    n_discard = 12)

# Run on a 4-D NIfTI volume
using NIfTI
Y = niread("path/to/bold.nii.gz")
ref_idx, t_vol, Y_masked, z_vol = analyze_and_plot(Y, params, "My recon label")

# Run on a multi-scale MAT file, pinning to the same slice
using MAT
vars = matread("path/to/mslr.mat")
X = vars["X"]
analyze_and_plot(X, params, Int(vars["Nscales"]), vars["patch_sizes"],
    "MSLR recon"; ref_slice_idx=ref_idx, threshold_quantile=0.99)
```

### Running tests

```julia
using Pkg
Pkg.test()
# or directly:
include("test/runtests.jl")
```

---

## Notes

- Complex-valued input arrays are automatically converted to magnitude (`abs.()`) before fitting; a warning is printed when this occurs.
- The analysis pipelines fit the GLM on brain voxels only. The brain mask is derived automatically via `bet_brain_mask` (FSL BET) and is not written to disk; for registration or surface analysis, use BET directly.
- Display thresholds in `analyze_and_plot` are percentile-based (top `threshold_quantile`, default 0.99, of |t| among brain voxels) rather than statistically corrected — they control plot coloring/titles only. The `fdr_correct` and `bonferroni_correct` functions perform actual Benjamini-Hochberg / Bonferroni correction and are available for standalone use on any returned t-map.
- The 5-D method of `analyze_and_plot` builds the GLM design matrix once and reuses it across all scales, then computes a percentile-based display threshold independently per scale. The t-score color scale and anatomical underlay are shared across scales to keep comparisons interpretable.
