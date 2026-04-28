using MAT, NIfTI, Revise
includet("../src/fmri_analysis.jl")
includet("../src/export.jl")

# ==============================================================================
# Experiment & GLM parameters
# ==============================================================================

# Tapping paradigm: alternating tap/rest blocks, 20 s each, 40 s period
params = ExperimentParams(
    tr=0.8f0,
    onsets=[collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
    durations=[fill(20.0f0, 9), fill(20.0f0, 9)],
    contrast=[1.0f0, -1.0f0, 0.0f0],
    n_discard=12)

# ==============================================================================
# Output directory
# ==============================================================================

const out_dir = "/mnt/storage/rexfung/20260409tap/recon/fsleyes"
mkpath(out_dir)

# ==============================================================================
# Script Execution
# ==============================================================================

# %% Load file: CAIPI + CG-SENSE
fn = "/mnt/storage/rexfung/20260409tap/recon/cgs/caipi_recon_cgs_l1_r5e-3.nii"
X = niread(fn)
cg_idx, caipi_cgs_tmap, caipi_cgs_Y = analyze_and_plot(X, params, "CAIPI sampling + CG-SENSE recon")
export_niftis(caipi_cgs_Y, caipi_cgs_tmap, "caipi_cgs", out_dir)

# %% Load file: time-shifted CAIPI + CG-SENSE
fn = "/mnt/storage/rexfung/20260409tap/recon/cgs/caipi_ts_recon_cgs_l1_r5e-3.nii"
X = niread(fn)
_, caipi_ts_cgs_tmap, caipi_ts_cgs_Y = analyze_and_plot(X, params, "time-shifted CAIPI sampling + CG-SENSE recon"; ref_slice_idx=cg_idx)
export_niftis(caipi_ts_cgs_Y, caipi_ts_cgs_tmap, "caipi_ts_cgs", out_dir)

# %% Load file: PD + CG-SENSE
fn = "/mnt/storage/rexfung/20260409tap/recon/cgs/pd_recon_cgs_l1_r5e-3.nii"
X = niread(fn)
_, pd_cgs_tmap, pd_cgs_Y = analyze_and_plot(X, params, "PD sampling + CG-SENSE recon"; ref_slice_idx=cg_idx)
export_niftis(pd_cgs_Y, pd_cgs_tmap, "pd_cgs", out_dir)

# %% Load file: CAIPI 3 scales
fn = "/mnt/storage/rexfung/20260409tap/recon/mslr/caipi_recon_3scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

X_sum = dropdims(sum(X, dims=5), dims=5)
caipi_3_idx, caipi_3_tmap, caipi_3_sum_Y = analyze_and_plot(X_sum, params,
    "CAIPI sampling + MSLR recon, $Nscales scales (sum)")
export_niftis(caipi_3_sum_Y, caipi_3_tmap, "caipi_$(Nscales)scales_sum", out_dir)

_, caipi_3_tmaps, caipi_3_Yvols = analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "CAIPI sampling + MSLR recon, $Nscales scales"; ref_slice_idx=caipi_3_idx)
export_niftis(caipi_3_Yvols, caipi_3_tmaps, patch_sizes, Nscales, "caipi", out_dir)

# %% Load file: time-shifted CAIPI 3 scales
fn = "/mnt/storage/rexfung/20260409tap/recon/mslr/caipi_ts_recon_3scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

X_sum = dropdims(sum(X, dims=5), dims=5)
caipi_ts_3_idx, caipi_ts_3_tmap, caipi_ts_3_sum_Y = analyze_and_plot(X_sum, params,
    "time-shifted CAIPI sampling + MSLR recon, $Nscales scales (sum)")
export_niftis(caipi_ts_3_sum_Y, caipi_ts_3_tmap, "caipi_ts_$(Nscales)scales_sum", out_dir)

_, caipi_ts_3_tmaps, caipi_ts_3_Yvols = analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "time-shifted CAIPI sampling + MSLR recon, $Nscales scales"; ref_slice_idx=caipi_ts_3_idx)
export_niftis(caipi_ts_3_Yvols, caipi_ts_3_tmaps, patch_sizes, Nscales, "caipi_ts", out_dir)

# %% Load file: PD 3 scales
fn = "/mnt/storage/rexfung/20260409tap/recon/mslr/pd_recon_3scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

X_sum = dropdims(sum(X, dims=5), dims=5)
pd_3_idx, pd_3_tmap, pd_3_sum_Y = analyze_and_plot(X_sum, params,
    "PD sampling + MSLR recon, $Nscales scales (sum)")
export_niftis(pd_3_sum_Y, pd_3_tmap, "pd_$(Nscales)scales_sum", out_dir)

_, pd_3_tmaps, pd_3_Yvols = analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "PD sampling + MSLR recon, $Nscales scales"; ref_slice_idx=pd_3_idx)
export_niftis(pd_3_Yvols, pd_3_tmaps, patch_sizes, Nscales, "pd", out_dir)

# %% Load file: CAIPI 1 scale LLR
fn = "/mnt/storage/rexfung/20260409tap/recon/mslr/caipi_recon_1scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

_, caipi_1_tmaps, caipi_1_Yvols = analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "CAIPI sampling + MSLR recon, $Nscales scales")
export_niftis(caipi_1_Yvols, caipi_1_tmaps, patch_sizes, Nscales, "caipi", out_dir)

# %% Load file: time-shifted CAIPI 1 scale LLR
fn = "/mnt/storage/rexfung/20260409tap/recon/mslr/caipi_ts_recon_1scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

_, caipi_ts_1_tmaps, caipi_ts_1_Yvols = analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "time-shifted CAIPI sampling + MSLR recon, $Nscales scales")
export_niftis(caipi_ts_1_Yvols, caipi_ts_1_tmaps, patch_sizes, Nscales, "caipi_ts", out_dir)

# %% Load file: PD 1 scale LLR
fn = "/mnt/storage/rexfung/20260409tap/recon/mslr/pd_recon_1scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

_, pd_1_tmaps, pd_1_Yvols = analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "PD sampling + MSLR recon, $Nscales scales")
export_niftis(pd_1_Yvols, pd_1_tmaps, patch_sizes, Nscales, "pd", out_dir)