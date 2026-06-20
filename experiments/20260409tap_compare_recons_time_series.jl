using MAT, NIfTI, Revise, Statistics
includet("../src/fmri_analysis.jl")
using .FmriAnalysis

# FSL environment — required for bet_brain_mask
let fsldir = get(ENV, "FSLDIR", expanduser("~/fsl"))
    ENV["FSLDIR"] = fsldir
    ENV["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    path_entries = split(get(ENV, "PATH", ""), ":")
    fsl_bin = joinpath(fsldir, "bin")
    fsl_bin ∈ path_entries || (ENV["PATH"] = fsl_bin * ":" * ENV["PATH"])
end

# %% ── Experiment & GLM parameters ───────────────────────────────────────────
params = ExperimentParams(
    tr        = 0.8f0,
    onsets    = [collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
    durations = [fill(20.0f0, 9), fill(20.0f0, 9)],
    contrast  = [1.0f0, -1.0f0, 0.0f0],
    n_discard = 0)

# %% ── Directories ─────────────────────────────────────────────────
basic_base = "/StorageRAID/rexfung/20260409tap/recon/basic"
mslr_base  = "/StorageRAID/rexfung/20260409tap/recon/mslr"

# %% ── Sampling schemes ───────────────────────────────────────────────────────
schemes = [
    ("pd_recon",       "PD sampling",                 "pd"),
    ("caipi_recon",    "CAIPI sampling",              "caipi"),
]

# %% Overview — same recons as the spatial comparison
recons = [
    (:mslr,  mslr_base,  "G+L+L_8xlambda", "3-scale low-rank"),
    (:mslr,  mslr_base,  "G+L_8xlambda",  "Global + local low-rank"),
    (:mslr,  mslr_base,  "L_8xlambda",  "Local low-rank"),
    (:basic, basic_base, "bart_l1_r0.0050_tv_r0.0050", "L1-wavelet + TV"),
    # (:basic, basic_base, "cgs_i100", "CG-SENSE"),
]
compare_recons_time_series(schemes, recons, params;
    save_dir = "plots", save_name = "overview_ts", stat = "z", normalize = "psc",
    condition_names = ["Tap", "Rest"], time_range = (0, 120),
    top_percent = 0.1)

# %% Global + 20^3 + 6^3, summed image — lambda sweep
recons = [
    (:mslr,  mslr_base,  "G+L+L_16xlambda", "3-scale (16xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_8xlambda",  "3-scale (8xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_4xlambda",  "3-scale (4xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_2xlambda",  "3-scale (2xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_1xlambda",  "3-scale (1xλ) sum"),
]
compare_recons_time_series(schemes, recons, params;
    save_dir = "plots", save_name = "gll_sum_ts", stat = "z",
    condition_names = ["Tap", "Rest"], time_range = (0, 120),
    top_percent = 0.1)

# %% Global + 20^3 + 6^3, 6^3 component — lambda sweep
recons = [
    (:mslr,  mslr_base,  "G+L+L_16xlambda", "3-scale (16xλ) [6,6,6]", 3),
    (:mslr,  mslr_base,  "G+L+L_8xlambda",  "3-scale (8xλ) [6,6,6]",  3),
    (:mslr,  mslr_base,  "G+L+L_4xlambda",  "3-scale (4xλ) [6,6,6]",  3),
    (:mslr,  mslr_base,  "G+L+L_2xlambda",  "3-scale (2xλ) [6,6,6]",  3),
    (:mslr,  mslr_base,  "G+L+L_1xlambda",  "3-scale (1xλ) [6,6,6]",  3),
]
compare_recons_time_series(schemes, recons, params;
    save_dir = "plots", save_name = "gll_scale3_ts", stat = "z",
    condition_names = ["Tap", "Rest"], time_range = (0, 120),
    top_percent = 0.1)
