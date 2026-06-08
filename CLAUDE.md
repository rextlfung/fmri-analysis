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

`src/fmri_analysis.jl` defines the `FmriAnalysis` module and, at the end of the file, `include`s `../scripts/run_analysis.jl` (the high-level analysis pipelines) and `src/export.jl` (NIfTI export helpers). There is no separate entry point — the three files together constitute the module. Both included files rely on their dependencies (`Statistics`, `NIfTI`, `Printf`, and the section 1–7 functions) being imported/defined by the parent module and must not add their own `using` statements. Note the unusual direction: a module under `src/` reaches up into `../scripts/` for `run_analysis.jl`.

### GLM pipeline

The pipeline operates in this order: `build_design_matrix` → `fit_glm` → `compute_tscores`. The `run_glm` wrapper accepts an optional `design_matrix` keyword argument; pass a pre-built matrix when calling it repeatedly with the same parameters (e.g. across MSLR scales) to avoid redundant FFT convolutions.

The GLM is fit on **brain voxels only**. The workflow is:
1. Flatten 4-D volume to `(n_scans × n_voxels)` matrix
2. Apply brain mask to select brain columns
3. Fit GLM on the masked subset
4. Reconstruct a full-size t-map by placing brain t-scores back into a zeros array

`analyze_and_plot` is a single function with two methods dispatched on input dimensionality (see `scripts/run_analysis.jl`):
- The **4-D method** `analyze_and_plot(X::…,4}, params, title_base; …)` returns three values: `(slice_idx, t_vol, Y_masked)`.
- The **5-D method** `analyze_and_plot(X::…,5}, params, Nscales, patch_sizes, title_base; …)` returns five values: `(slice_idx, t_vols, Y_vols, t_sum_vol, Y_sum_masked)`.

Destructure exactly the number of values the relevant method returns — assigning to a single variable captures a tuple, which will fail when used as a `ref_slice_idx`.

### Brain masking

Brain masking shells out to FSL `bet` via `bet_brain_mask`. The function writes a temporary NIfTI to `/tmp`, calls `bet`, reads the mask back, then cleans up. There is no pure-Julia fallback.

Three environment variables must be set before running any analysis:
- `FSLDIR` — path to the FSL installation root (e.g. `/home/user/fsl`)
- `PATH` — must include `$FSLDIR/bin` so `bet` is found
- `FSLOUTPUTTYPE=NIFTI_GZ` — `bet_brain_mask` reads `*_mask.nii.gz`; other output types will cause a file-not-found error

To run a batch script from the shell: `FSLDIR=... FSLOUTPUTTYPE=NIFTI_GZ PATH="$FSLDIR/bin:$PATH" julia --project=.. <script>.jl`

### MSLR data format

Standard reconstructions are 4-D `(nx, ny, nz, nt)`. MSLR reconstructions are 5-D `(nx, ny, nz, nt, Nscales)`. The 5-D method of `analyze_and_plot` handles the 5-D case: it builds one shared brain mask from the temporal mean of the summed reconstruction, builds the design matrix once, then loops over scales.

### Experiment scripts

Files in `experiments/` are named by session date (e.g. `20260409tap.jl`) and are structured for cell-by-cell execution in VS Code with the Julia extension (`# %%` cell markers). They are not importable modules. Each script sets `params` (an `ExperimentParams`) near the top; the first reconstruction analyzed in a session establishes a `ref_slice_idx` that is passed to all subsequent calls to keep plots at the same anatomical location.
