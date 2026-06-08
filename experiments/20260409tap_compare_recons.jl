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

# %% ── Sampling schemes ───────────────────────────────────────────────────────
# 3-tuple: (file_base, display_label, export_prefix)
# compare_recons uses the first two elements; the third is ignored.

schemes = [
    ("pd_recon",       "PD sampling",                 "pd"),
    ("caipi_ts_recon", "time-shifted CAIPI sampling", "caipi_ts"),
    ("caipi_recon",    "CAIPI sampling",              "caipi"),
]

# %% ── Reconstruction methods ─────────────────────────────────────────────────
# 4-tuple: (:basic | :mslr, base_dir, identifier, display_label)
# 5-tuple: (:mslr,          base_dir, identifier, display_label, scale_n::Int)
#   :basic  → $(base_dir)/$(file_base)_$(identifier).mat,  key "img" (4-D)
#   :mslr   → $(base_dir)/$(identifier)/$(file_base).mat,  key "X"  (5-D)
#             4-tuple: sums all scales
#             5-tuple: extracts the n-th scale (1-based)
#             patch_sizes for G+L+L configs: scale 1=[90,90,60]  2=[20,20,20]  3=[6,6,6]

basic_base = "/StorageRAID/rexfung/20260409tap/recon/basic"
mslr_base  = "/StorageRAID/rexfung/20260409tap/recon/mslr"

recons = [
    (:mslr,  mslr_base,  "G+L+L_5xlambda",             "MSLR (5xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_4xlambda",             "MSLR (4xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_3xlambda",             "MSLR (3xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_2xlambda",             "MSLR (2xλ) sum"),
    (:basic, basic_base, "bart_l1_r0.0050_tv_r0.0050", "BART (L1+TV)"),
]

# %% ── Run ────────────────────────────────────────────────────────────────────
compare_recons(schemes, recons, params)

# %%
recons = [
    (:mslr,  mslr_base,  "G+L+L_5xlambda",             "MSLR (5xλ) [6,6,6]",   3),
    (:mslr,  mslr_base,  "G+L+L_4xlambda",             "MSLR (4xλ) [6,6,6]",   3),
    (:mslr,  mslr_base,  "G+L+L_3xlambda",             "MSLR (3xλ) [6,6,6]",   3),
    (:mslr,  mslr_base,  "G+L+L_2xlambda",             "MSLR (2xλ) [6,6,6]",   3),
    (:basic, basic_base, "bart_l1_r0.0050_tv_r0.0050", "BART (L1+TV)"),
]

# %% ── Run ────────────────────────────────────────────────────────────────────
compare_recons(schemes, recons, params)