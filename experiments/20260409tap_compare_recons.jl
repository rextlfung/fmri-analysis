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
# 3-tuple: (file_base, display_label, export_prefix)
# compare_recons uses the first two elements; the third is ignored.
schemes = [
    ("pd_recon",       "PD sampling",                 "pd"),
    # ("caipi_ts_recon", "time-shifted CAIPI sampling", "caipi_ts"),
    ("caipi_recon",    "CAIPI sampling",              "caipi"),
]

# %% Overview
recons = [
    (:mslr,  mslr_base,  "G+L+L_8xlambda", "3-scale low-rank"),
    (:mslr,  mslr_base,  "G+L_8xlambda",  "Global + local low-rank"),
    (:mslr,  mslr_base,  "L_8xlambda",  "Local low-rank"),
    (:basic, basic_base, "bart_l1_r0.0050_tv_r0.0050", "L1-wavelet + TV"),
    (:basic, basic_base, "cgs_i100", "CG-SENSE"),
]
ref_si = compare_recons(schemes, recons, params;
    save_dir = "plots", save_name = "overview", stat = "z")

# %% Global + 20^3 + 6^3, summed image
recons = [
    (:mslr,  mslr_base,  "G+L+L_16xlambda", "3-scale low-rank (16xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_8xlambda", "3-scale low-rank (8xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_4xlambda", "3-scale low-rank (4xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_2xlambda", "3-scale low-rank (2xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_1xlambda", "3-scale low-rank (1xλ) sum"),
]
compare_recons(schemes, recons, params; slice_indices=ref_si,
    save_dir = "plots", save_name = "gll_sum", stat = "z")

# %% Global + 20^3 + 6^3, 6^3 component
recons = [
    (:mslr,  mslr_base,  "G+L+L_16xlambda", "3-scale low-rank (16xλ) [6,6,6]", 3),
    (:mslr,  mslr_base,  "G+L+L_8xlambda", "3-scale low-rank (8xλ) [6,6,6]", 3),
    (:mslr,  mslr_base,  "G+L+L_4xlambda", "3-scale low-rank (4xλ) [6,6,6]", 3),
    (:mslr,  mslr_base,  "G+L+L_2xlambda", "3-scale low-rank (2xλ) [6,6,6]", 3),
    (:mslr,  mslr_base,  "G+L+L_1xlambda", "3-scale low-rank (1xλ) [6,6,6]", 3),
]
compare_recons(schemes, recons, params; slice_indices=ref_si,
    save_dir = "plots", save_name = "gll_scale3", stat = "z")

# %% Global + Local, summed image
recons = [
    (:mslr,  mslr_base,  "G+L_16xlambda", "Global + local low-rank (16xλ) sum"),
    (:mslr,  mslr_base,  "G+L_8xlambda",  "Global + local low-rank (8xλ) sum"),
    (:mslr,  mslr_base,  "G+L_4xlambda",  "Global + local low-rank (4xλ) sum"),
    (:mslr,  mslr_base,  "G+L_2xlambda",  "Global + local low-rank (2xλ) sum"),
    (:mslr,  mslr_base,  "G+L_1xlambda",  "Global + local low-rank (1xλ) sum"),
]
compare_recons(schemes, recons, params; slice_indices=ref_si,
    save_dir = "plots", save_name = "gl_sum", stat = "z")

# %% Global + Local, local component (scale 2)
recons = [
    (:mslr,  mslr_base,  "G+L_16xlambda", "Global + local low-rank (16xλ) local", 2),
    (:mslr,  mslr_base,  "G+L_8xlambda",  "Global + local low-rank (8xλ) local",  2),
    (:mslr,  mslr_base,  "G+L_4xlambda",  "Global + local low-rank (4xλ) local",  2),
    (:mslr,  mslr_base,  "G+L_2xlambda",  "Global + local low-rank (2xλ) local",  2),
    (:mslr,  mslr_base,  "G+L_1xlambda",  "Global + local low-rank (1xλ) local",  2),
]
compare_recons(schemes, recons, params; slice_indices=ref_si,
    save_dir = "plots", save_name = "gl_local", stat = "z")

# %% Local only, summed image
recons = [
    (:mslr,  mslr_base,  "L_16xlambda", "Local low-rank (16xλ)"),
    (:mslr,  mslr_base,  "L_8xlambda",  "Local low-rank (8xλ)"),
    (:mslr,  mslr_base,  "L_4xlambda",  "Local low-rank (4xλ)"),
    (:mslr,  mslr_base,  "L_2xlambda",  "Local low-rank (2xλ)"),
    (:mslr,  mslr_base,  "L_1xlambda",  "Local low-rank (1xλ)"),
]
compare_recons(schemes, recons, params; slice_indices=ref_si,
    save_dir = "plots", save_name = "llr_sum", stat = "z")
