using MAT, NIfTI, Revise
includet("../src/fmri_analysis.jl")
using .FmriTscores

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

const base_dir = "/mnt/storage/rexfung/20260409tap/recon"
const out_dir  = "$base_dir/fsleyes"
mkpath(out_dir)

# ==============================================================================
# NIfTI reconstructions (CG-SENSE variants)
# ==============================================================================

# (path, label, export_prefix)
# The first entry establishes the shared reference slice index.
nifti_recons = [
    ("$base_dir/cgs/caipi_recon_cgs_l1_r5e-3.nii",    "CAIPI sampling + CG-SENSE recon",             "caipi_cgs"),
    ("$base_dir/cgs/caipi_ts_recon_cgs_l1_r5e-3.nii", "time-shifted CAIPI sampling + CG-SENSE recon", "caipi_ts_cgs"),
    ("$base_dir/cgs/pd_recon_cgs_l1_r5e-3.nii",       "PD sampling + CG-SENSE recon",                 "pd_cgs"),
]

cg_idx = nothing
for (fn, label, prefix) in nifti_recons
    idx, tmap, Y = analyze_and_plot(niread(fn), params, label; ref_slice_idx=cg_idx)
    isnothing(cg_idx) && (cg_idx = idx)
    export_niftis(Y, tmap, prefix, out_dir)
end

# ==============================================================================
# MSLR reconstructions
# ==============================================================================

# (path, label, export_prefix, include_sum)
# include_sum=true  → analyze the summed reconstruction first and use it to pin
#                     the reference slice; also exports the summed t-map.
# include_sum=false → let analyze_and_plot_mslr pick the peak slice internally
#                     (used for single-scale LLR where the sum is trivial).
mslr_recons = [
    ("$base_dir/mslr/caipi_recon_3scales.mat",    "CAIPI sampling + MSLR recon",             "caipi",    true),
    ("$base_dir/mslr/caipi_ts_recon_3scales.mat", "time-shifted CAIPI sampling + MSLR recon", "caipi_ts", true),
    ("$base_dir/mslr/pd_recon_3scales.mat",       "PD sampling + MSLR recon",                 "pd",       true),
    ("$base_dir/mslr/caipi_recon_1scales.mat",    "CAIPI sampling + MSLR recon",             "caipi",    false),
    ("$base_dir/mslr/caipi_ts_recon_1scales.mat", "time-shifted CAIPI sampling + MSLR recon", "caipi_ts", false),
    ("$base_dir/mslr/pd_recon_1scales.mat",       "PD sampling + MSLR recon",                 "pd",       false),
]

for (fn, label, prefix, include_sum) in mslr_recons
    vars = matread(fn)
    X = vars["X"]
    Nscales = Int(vars["Nscales"])
    patch_sizes = vars["patch_sizes"]

    ref_idx = nothing
    if include_sum
        X_sum = dropdims(sum(X, dims=5), dims=5)
        ref_idx, tmap, Y = analyze_and_plot(X_sum, params, "$label, $Nscales scales (sum)")
        export_niftis(Y, tmap, "$(prefix)_$(Nscales)scales_sum", out_dir)
    end

    _, tmaps, Yvols = analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
        "$label, $Nscales scales"; ref_slice_idx=ref_idx)
    export_niftis(Yvols, tmaps, patch_sizes, Nscales, prefix, out_dir)
end
