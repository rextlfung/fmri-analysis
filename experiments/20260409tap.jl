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

basic_recon_methods = [
    ("rss",                        "RSS",          "rss"),
    ("cgs_i100",                   "CG-SENSE",     "cgs"),
    ("bart_l1_r0.0050_tv_r0.0050", "BART (L1+TV)", "bart"),
]

basic_base = "/StorageRAID/rexfung/20260409tap/recon/basic"

mslr_cfgs = [
    "G+L+L_5xlambda",
]

schemes = [
    ("pd_recon",       "PD sampling",                 "pd"),
    ("caipi_ts_recon", "time-shifted CAIPI sampling", "caipi_ts"),
    ("caipi_recon",    "CAIPI sampling",              "caipi"),
]

mslr_base = "/StorageRAID/rexfung/20260409tap/recon/mslr"

params = ExperimentParams(
    tr=0.8f0,
    onsets=[collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
    durations=[fill(20.0f0, 9), fill(20.0f0, 9)],
    contrast=[1.0f0, -1.0f0, 0.0f0],
    n_discard=0)

# ── Basic reconstructions ──────────────────────────────────────────────────────
let
    basic_out = joinpath(basic_base, "fsleyes")
    mkpath(basic_out)
    basic_ref_idx = nothing

    # Brain mask and design matrix are shared: all basic recons share the same
    # anatomy and scan timing.
    probe_Y = matread(joinpath(basic_base, "$(schemes[1][1])_$(basic_recon_methods[1][1]).mat"))["img"]
    probe_Y = probe_Y[:, :, :, (params.n_discard+1):end]
    basic_mask = bet_brain_mask(dropdims(mean(Float32.(probe_Y), dims=4), dims=4))
    design_mat = build_design_matrix(params.onsets, params.durations, size(probe_Y, 4), params.tr)

    for (scheme_base, scheme_label, scheme_prefix) in schemes
        for (recon_suffix, recon_label, recon_prefix) in basic_recon_methods
            fn = joinpath(basic_base, "$(scheme_base)_$(recon_suffix).mat")
            vars = matread(fn)
            X = vars["img"]
            label = "$scheme_label + $recon_label"

            slice_idx, t_vol, Y_masked = analyze_and_plot(
                X, params, label; ref_slice_idx=basic_ref_idx,
                brain_mask=basic_mask, design_matrix=design_mat)
            isnothing(basic_ref_idx) && (basic_ref_idx = slice_idx)
            export_niftis(Y_masked, t_vol, "$(scheme_prefix)_$(recon_prefix)", basic_out)
        end
    end
end

# ── MSLR reconstructions ───────────────────────────────────────────────────────
for folder in mslr_cfgs
    cfg_out = joinpath(mslr_base, folder, "fsleyes")
    mkpath(cfg_out)

    # ref_slice_idx pinned from the first dataset's summed reconstruction and
    # reused across all three sampling schemes so all plots show the same slice.
    cfg_ref_idx = nothing

    for (file_base, scheme_label, export_prefix) in schemes
        fn = joinpath(mslr_base, folder, "$(file_base).mat")
        vars = matread(fn)
        X           = vars["X"]
        Nscales     = Int(only(vars["Nscales"]))
        patch_sizes = vars["patch_sizes"]
        label = "$scheme_label ($folder)"

        slice_idx, tmaps, Yvols, t_vol_sum, Y_masked_sum = analyze_and_plot_mslr(
            X, params, Nscales, patch_sizes,
            "$label, "; ref_slice_idx=cfg_ref_idx, plot_sum=true)
        isnothing(cfg_ref_idx) && (cfg_ref_idx = slice_idx)
        if Nscales > 1
            export_niftis(Y_masked_sum, t_vol_sum, "$(export_prefix)_$(Nscales)scales_sum", cfg_out)
        end
        export_niftis(Yvols, tmaps, patch_sizes, Nscales, export_prefix, cfg_out)
    end
end