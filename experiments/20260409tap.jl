using MAT, NIfTI, Revise
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

# ==============================================================================
# MSLR reconstructions — regularisation comparison (5 configurations)
# ==============================================================================
# Compares activation maps across 5 regularisation configurations for the
# 20260409 tapping session, all at tol=1e-3:
#   - GLR        (Nscales=1, global low-rank)
#   - LLR        (Nscales=1, locally low-rank)
#   - G+LLR      (Nscales=2, global + local low-rank)
#   - G+VLR      (Nscales=2, global + voxelwise low-rank)
#   - G+L+VLR    (Nscales=3, global + local + voxelwise low-rank)
# Each configuration contains 3 datasets (CAIPI / time-shifted CAIPI / PD
# sampling); plots within a configuration are pinned to the same anatomical
# slice. All folders contain nt=375 frames (pre-trimmed) → n_discard=0.
# ==============================================================================

mslr_cfgs = [
    "GLR_100itrs_tol=1e-3",
    "LLR_100itrs_tol=1e-3",
    "G+LLR_100itrs_tol=1e-3",
    "G+VLR_100itrs_tol=1e-3",
    "G+L+VLR_100itrs_tol=1e-3",
]

mslr_schemes = [
    ("caipi_recon",    "CAIPI sampling",              "caipi"),
    ("caipi_ts_recon", "time-shifted CAIPI sampling", "caipi_ts"),
    ("pd_recon",       "PD sampling",                 "pd"),
]

mslr_base = "/StorageRAID/rexfung/20260409tap/recon/mslr"

params = ExperimentParams(
    tr=0.8f0,
    onsets=[collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
    durations=[fill(20.0f0, 9), fill(20.0f0, 9)],
    contrast=[1.0f0, -1.0f0, 0.0f0],
    n_discard=0)

for folder in mslr_cfgs
    cfg_out = joinpath(mslr_base, folder, "fsleyes")
    mkpath(cfg_out)

    # ref_slice_idx pinned from the first dataset's summed reconstruction and
    # reused across all three sampling schemes so all plots show the same slice.
    cfg_ref_idx = nothing

    for (file_base, scheme_label, export_prefix) in mslr_schemes
        fn = joinpath(mslr_base, folder, "$(file_base).mat")
        vars = matread(fn)
        X           = vars["X"]
        Nscales     = Int(only(vars["Nscales"]))
        patch_sizes = vars["patch_sizes"]
        label = "$scheme_label ($folder)"

        slice_idx, tmaps, Yvols, t_vol_sum, Y_masked_sum = analyze_and_plot_mslr(
            X, params, Nscales, patch_sizes,
            "$label, $Nscales scales"; ref_slice_idx=cfg_ref_idx, plot_sum=true, q=0.01)
        isnothing(cfg_ref_idx) && (cfg_ref_idx = slice_idx)
        if Nscales > 1
            export_niftis(Y_masked_sum, t_vol_sum, "$(export_prefix)_$(Nscales)scales_sum", cfg_out)
        end
        export_niftis(Yvols, tmaps, patch_sizes, Nscales, export_prefix, cfg_out)
    end
end
