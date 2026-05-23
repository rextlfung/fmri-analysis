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
# MSLR reconstructions — convergence / normalisation comparison (4 configurations)
# ==============================================================================
# Compares activation maps across 4 MSLR configurations (differing in iteration
# count, stopping criterion, and normalisation) for the 20260409 tapping session.
# The 3 datasets (CAIPI / time-shifted CAIPI / PD sampling) are analysed within
# each configuration; plots are pinned to the same anatomical slice.
#
# Each entry: (subfolder, file_suffix, n_discard)
# Folders with nt=375 were reconstructed from pre-trimmed data → n_discard=0.
# Folders with nt=387 retain the leading instructional frames → n_discard=12.
# ==============================================================================

mslr_cfgs = [
    ("100itrs_tol=1e-2",                "",                    0),
]

mslr_schemes = [
    ("caipi_recon",    "CAIPI sampling",              "caipi"),
    ("caipi_ts_recon", "time-shifted CAIPI sampling", "caipi_ts"),
    ("pd_recon",       "PD sampling",                 "pd"),
]

mslr_base = "/StorageRAID/rexfung/20260409tap/recon/mslr"

for (folder, suffix, n_discard_cfg) in mslr_cfgs
    cfg_params = ExperimentParams(
        tr=0.8f0,
        onsets=[collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
        durations=[fill(20.0f0, 9), fill(20.0f0, 9)],
        contrast=[1.0f0, -1.0f0, 0.0f0],
        n_discard=n_discard_cfg)

    cfg_out = joinpath(mslr_base, folder, "fsleyes")
    mkpath(cfg_out)

    # ref_slice_idx pinned from the first dataset's summed reconstruction and
    # reused across all three sampling schemes so all plots show the same slice.
    cfg_ref_idx = nothing

    for (file_base, scheme_label, export_prefix) in mslr_schemes
        fn = joinpath(mslr_base, folder, "$(file_base)$(suffix).mat")
        vars = matread(fn)
        X           = vars["X"]
        Nscales     = Int(vars["Nscales"])
        patch_sizes = vars["patch_sizes"]
        label = "$scheme_label + MSLR recon ($folder)"

        X_sum = dropdims(sum(X, dims=5), dims=5)
        slice_idx, tmap, Y = analyze_and_plot(X_sum, cfg_params,
            "$label, $Nscales scales (sum)"; ref_slice_idx=cfg_ref_idx)
        isnothing(cfg_ref_idx) && (cfg_ref_idx = slice_idx)
        export_niftis(Y, tmap, "$(export_prefix)_$(Nscales)scales_sum", cfg_out)

        _, tmaps, Yvols = analyze_and_plot_mslr(X, cfg_params, Nscales, patch_sizes,
            "$label, $Nscales scales"; ref_slice_idx=cfg_ref_idx)
        export_niftis(Yvols, tmaps, patch_sizes, Nscales, export_prefix, cfg_out)
    end
end
