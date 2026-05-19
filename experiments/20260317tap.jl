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

const base_dir = "/mnt/storage/rexfung/20260317tap"

# ==============================================================================
# NIfTI reconstructions
# ==============================================================================

# (path, label, flip_dim1)
nifti_recons = [
    ("$base_dir/prod/smsepi.nii.gz",      "SMS-EPI + slice-GRAPPA recon",    true),
    ("$base_dir/recon/caipi_recon_cgs.nii", "CAIPI sampling + CG-SENSE recon", false),
    ("$base_dir/recon/pd_recon_cgs.nii",    "PD sampling + CG-SENSE recon",    false),
]

for (fn, label, flip) in nifti_recons
    Y = flip ? reverse(niread(fn), dims=1) : niread(fn)
    analyze_and_plot(Y, params, label)
end

# ==============================================================================
# MSLR reconstructions
# ==============================================================================

# (path, label) — Nscales and patch_sizes are read from each .mat file
mslr_recons = [
    ("$base_dir/recon/caipi_recon_5scales.mat", "CAIPI sampling + MSLR recon"),
    ("$base_dir/recon/pd_recon_5scales.mat",    "Poisson-disc random sampling + MSLR recon"),
    ("$base_dir/recon/caipi_recon_4scales.mat", "CAIPI sampling + MSLR recon"),
    ("$base_dir/recon/pd_recon_4scales.mat",    "Poisson-disc random sampling + MSLR recon"),
    ("$base_dir/recon/caipi_recon_3scales.mat", "CAIPI sampling + MSLR recon"),
    ("$base_dir/recon/pd_recon_3scales.mat",    "Poisson-disc random sampling + MSLR recon"),
]

for (fn, label) in mslr_recons
    vars = matread(fn)
    X = vars["X"]
    Nscales = Int(vars["Nscales"])
    patch_sizes = vars["patch_sizes"]
    idx, _, _ = analyze_and_plot(dropdims(sum(X, dims=5), dims=5), params,
        "$label, $Nscales scales (sum)")
    analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
        "$label, $Nscales scales"; ref_slice_idx=idx)
end
