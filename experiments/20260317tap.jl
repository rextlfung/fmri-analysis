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
    n_discard = 12)

# %% ── Directories ───────────────────────────────────────────────────────────
basic_base = "/StorageRAID/rexfung/20260317tap/recon/basic"
mslr_base  = "/StorageRAID/rexfung/20260317tap/recon/mslr"

# ═══════════════════════════════════════════════════════════════════════════════
# Per-recon analyze_and_plot + NIfTI export
# ═══════════════════════════════════════════════════════════════════════════════

# %% ── MSLR reconstructions ──────────────────────────────────────────────────
mslr_cfgs = [
    "G+L+L_8xlambda",
    "G+L+L_6xlambda",
    "G+L+L_4xlambda",
    "G+L+L_2xlambda",
    "G+L+L_1xlambda",
]

schemes_all = [
    ("pd_recon",    "PD sampling",    "pd"),
    ("caipi_recon", "CAIPI sampling", "caipi"),
]

ref_idx = nothing

for folder in mslr_cfgs
    cfg_out = joinpath(mslr_base, folder, "fsleyes")
    mkpath(cfg_out)

    for (file_base, scheme_label, export_prefix) in schemes_all
        fn = joinpath(mslr_base, folder, "$(file_base).mat")
        vars = matread(fn)
        X           = vars["X"]
        Nscales     = Int(only(vars["Nscales"]))
        patch_sizes = vars["patch_sizes"]
        label = "$scheme_label ($folder)"

        slice_idx, tmaps, Yvols, t_vol_sum, Y_masked_sum = analyze_and_plot(
            X, params, Nscales, patch_sizes, label;
            ref_slice_idx=ref_idx, plot_sum=true)
        isnothing(ref_idx) && (global ref_idx = slice_idx)
        if Nscales > 1
            export_niftis(Y_masked_sum, t_vol_sum, "$(export_prefix)_$(Nscales)scales_sum", cfg_out)
        end
        export_niftis(Yvols, tmaps, patch_sizes, Nscales, export_prefix, cfg_out)
    end
end

# %% ── Basic reconstructions ─────────────────────────────────────────────────
basic_recon_methods = [
    ("cgs_i100",                   "CG-SENSE",     "cgs"),
    ("bart_l1_r0.0050_tv_r0.0050", "BART (L1+TV)", "bart"),
]

basic_out = joinpath(basic_base, "fsleyes")
mkpath(basic_out)

probe_Y = matread(joinpath(basic_base, "$(schemes_all[1][1])_$(basic_recon_methods[1][1]).mat"))["img"]
probe_Y = probe_Y[:, :, :, (params.n_discard+1):end]
basic_mask = bet_brain_mask(dropdims(mean(Float32.(abs.(probe_Y)), dims=4), dims=4))
design_mat = build_design_matrix(params.onsets, params.durations, size(probe_Y, 4), params.tr)

for (scheme_base, scheme_label, scheme_prefix) in schemes_all
    for (recon_suffix, recon_label, recon_prefix) in basic_recon_methods
        fn = joinpath(basic_base, "$(scheme_base)_$(recon_suffix).mat")
        vars = matread(fn)
        X = vars["img"]
        label = "$scheme_label + $recon_label"

        slice_idx, t_vol, Y_masked = analyze_and_plot(
            X, params, label;
            ref_slice_idx=ref_idx, brain_mask=basic_mask, design_matrix=design_mat)
        isnothing(ref_idx) && (global ref_idx = slice_idx)
        export_niftis(Y_masked, t_vol, "$(scheme_prefix)_$(recon_prefix)", basic_out)
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# Spatial comparison (compare_recons)
# ═══════════════════════════════════════════════════════════════════════════════

# %% ── Schemes for comparison ────────────────────────────────────────────────
schemes_cmp = [
    ("pd_recon",    "PD sampling",    "pd"),
    ("caipi_recon", "CAIPI sampling", "caipi"),
]

# %% Overview
recons = [
    (:mslr,  mslr_base,  "G+L+L_8xlambda", "3-scale low-rank"),
    (:mslr,  mslr_base,  "G+L_8xlambda",  "Global + local low-rank"),
    (:mslr,  mslr_base,  "L_8xlambda",  "Local low-rank"),
    (:basic, basic_base, "bart_l1_r0.0050_tv_r0.0050", "L1-wavelet + TV"),
    (:basic, basic_base, "cgs_i100", "CG-SENSE"),
]
ref_si = compare_recons(schemes_cmp, recons, params;
    save_dir = "plots", save_name = "overview", stat = "z")

# %% Global + 20^3 + 6^3, summed image
recons = [
    (:mslr,  mslr_base,  "G+L+L_10xlambda", "3-scale low-rank (10xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_8xlambda",  "3-scale low-rank (8xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_6xlambda",  "3-scale low-rank (6xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_4xlambda",  "3-scale low-rank (4xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_2xlambda",  "3-scale low-rank (2xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_1xlambda",  "3-scale low-rank (1xλ) sum"),
]
compare_recons(schemes_cmp, recons, params; slice_indices=ref_si,
    save_dir = "plots", save_name = "gll_sum", stat = "z")

# %% Global + 20^3 + 6^3, 6^3 component
recons = [
    (:mslr,  mslr_base,  "G+L+L_10xlambda", "3-scale low-rank (10xλ) [6,6,6]", 3),
    (:mslr,  mslr_base,  "G+L+L_8xlambda",  "3-scale low-rank (8xλ) [6,6,6]",  3),
    (:mslr,  mslr_base,  "G+L+L_6xlambda",  "3-scale low-rank (6xλ) [6,6,6]",  3),
    (:mslr,  mslr_base,  "G+L+L_4xlambda",  "3-scale low-rank (4xλ) [6,6,6]",  3),
    (:mslr,  mslr_base,  "G+L+L_2xlambda",  "3-scale low-rank (2xλ) [6,6,6]",  3),
    (:mslr,  mslr_base,  "G+L+L_1xlambda",  "3-scale low-rank (1xλ) [6,6,6]",  3),
]
compare_recons(schemes_cmp, recons, params; slice_indices=ref_si,
    save_dir = "plots", save_name = "gll_scale3", stat = "z")

# %% Global + Local, summed image
recons = [
    (:mslr,  mslr_base,  "G+L_10xlambda", "Global + local low-rank (10xλ) sum"),
    (:mslr,  mslr_base,  "G+L_8xlambda",  "Global + local low-rank (8xλ) sum"),
    (:mslr,  mslr_base,  "G+L_6xlambda",  "Global + local low-rank (6xλ) sum"),
    (:mslr,  mslr_base,  "G+L_4xlambda",  "Global + local low-rank (4xλ) sum"),
    (:mslr,  mslr_base,  "G+L_2xlambda",  "Global + local low-rank (2xλ) sum"),
    (:mslr,  mslr_base,  "G+L_1xlambda",  "Global + local low-rank (1xλ) sum"),
]
compare_recons(schemes_cmp, recons, params; slice_indices=ref_si,
    save_dir = "plots", save_name = "gl_sum", stat = "z")

# %% Global + Local, local component (scale 2)
recons = [
    (:mslr,  mslr_base,  "G+L_10xlambda", "Global + local low-rank (10xλ) local", 2),
    (:mslr,  mslr_base,  "G+L_8xlambda",  "Global + local low-rank (8xλ) local",  2),
    (:mslr,  mslr_base,  "G+L_6xlambda",  "Global + local low-rank (6xλ) local",  2),
    (:mslr,  mslr_base,  "G+L_4xlambda",  "Global + local low-rank (4xλ) local",  2),
    (:mslr,  mslr_base,  "G+L_2xlambda",  "Global + local low-rank (2xλ) local",  2),
    (:mslr,  mslr_base,  "G+L_1xlambda",  "Global + local low-rank (1xλ) local",  2),
]
compare_recons(schemes_cmp, recons, params; slice_indices=ref_si,
    save_dir = "plots", save_name = "gl_local", stat = "z")

# %% Local only, summed image
recons = [
    (:mslr,  mslr_base,  "L_10xlambda", "Local low-rank (10xλ)"),
    (:mslr,  mslr_base,  "L_8xlambda",  "Local low-rank (8xλ)"),
    (:mslr,  mslr_base,  "L_6xlambda",  "Local low-rank (6xλ)"),
    (:mslr,  mslr_base,  "L_4xlambda",  "Local low-rank (4xλ)"),
    (:mslr,  mslr_base,  "L_2xlambda",  "Local low-rank (2xλ)"),
    (:mslr,  mslr_base,  "L_1xlambda",  "Local low-rank (1xλ)"),
]
compare_recons(schemes_cmp, recons, params; slice_indices=ref_si,
    save_dir = "plots", save_name = "llr_sum", stat = "z")


# ═══════════════════════════════════════════════════════════════════════════════
# Time-series comparison (compare_recons_time_series)
# ═══════════════════════════════════════════════════════════════════════════════

# %% Overview
recons = [
    (:mslr,  mslr_base,  "G+L+L_8xlambda", "3-scale low-rank"),
    (:mslr,  mslr_base,  "G+L_8xlambda",  "Global + local low-rank"),
    (:mslr,  mslr_base,  "L_8xlambda",  "Local low-rank"),
    (:basic, basic_base, "bart_l1_r0.0050_tv_r0.0050", "L1-wavelet + TV"),
]
compare_recons_time_series(schemes_cmp, recons, params;
    save_dir = "plots", save_name = "overview_ts", stat = "z", normalize = "psc",
    condition_names = ["Tap", "Rest"], time_range = (0, 120),
    top_percent = 0.1)

# %% Global + 20^3 + 6^3, summed image — lambda sweep
recons = [
    (:mslr,  mslr_base,  "G+L+L_10xlambda", "3-scale (10xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_8xlambda",  "3-scale (8xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_6xlambda",  "3-scale (6xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_4xlambda",  "3-scale (4xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_2xlambda",  "3-scale (2xλ) sum"),
    (:mslr,  mslr_base,  "G+L+L_1xlambda",  "3-scale (1xλ) sum"),
]
compare_recons_time_series(schemes_cmp, recons, params;
    save_dir = "plots", save_name = "gll_sum_ts", stat = "z",
    condition_names = ["Tap", "Rest"], time_range = (0, 120),
    top_percent = 0.1)

# %% Global + 20^3 + 6^3, 6^3 component — lambda sweep
recons = [
    (:mslr,  mslr_base,  "G+L+L_10xlambda", "3-scale (10xλ) [6,6,6]", 3),
    (:mslr,  mslr_base,  "G+L+L_8xlambda",  "3-scale (8xλ) [6,6,6]",  3),
    (:mslr,  mslr_base,  "G+L+L_6xlambda",  "3-scale (6xλ) [6,6,6]",  3),
    (:mslr,  mslr_base,  "G+L+L_4xlambda",  "3-scale (4xλ) [6,6,6]",  3),
    (:mslr,  mslr_base,  "G+L+L_2xlambda",  "3-scale (2xλ) [6,6,6]",  3),
    (:mslr,  mslr_base,  "G+L+L_1xlambda",  "3-scale (1xλ) [6,6,6]",  3),
]
compare_recons_time_series(schemes_cmp, recons, params;
    save_dir = "plots", save_name = "gll_scale3_ts", stat = "z",
    condition_names = ["Tap", "Rest"], time_range = (0, 120),
    top_percent = 0.1)
