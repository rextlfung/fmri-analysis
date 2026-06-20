# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction Comparison — Time Series
# ─────────────────────────────────────────────────────────────────────────────
#
# Defines `compare_recons_time_series`, a companion to `compare_recons` that
# plots voxel-level time series across reconstructions rather than spatial maps.
#
# Included into the `FmriAnalysis` module by `src/fmri_analysis.jl`; relies on
# `Statistics`, `CairoMakie`, `MAT`, and sections 1–7 of the parent module.
# Must not add its own `using` statements.

"""
    compare_recons_time_series(schemes, recons, params; ...)

For each sampling scheme, run the GLM on every reconstruction and produce two
time-series comparison figures:

1. **Peak voxel** — the BOLD signal at the single most active voxel (highest
   positive stat score), with one line per reconstruction.
2. **Top-n% average** — the mean BOLD signal across the top `top_percent`
   voxels with positive stat scores, with ±1 SEM shading.

Task blocks (from `params.onsets` / `params.durations`) are shaded in the
background with per-condition colors, and the fitted GLM model for each
reconstruction is overlaid as a dashed line.

# Arguments
- `schemes` : same format as `compare_recons` — vector of 3-tuples
              `(file_base, display_label, _)`.
- `recons`  : same format as `compare_recons` — vector of tuples
              `(type, base_dir, identifier, display_label[, scale_n])`.
- `params`  : `ExperimentParams` with scan timing and GLM settings.

# Keyword arguments
- `stat`        : `"t"` or `"z"` — which statistic to rank voxels by (default `"t"`).
                  Rankings are identical (monotonic transform, shared df).
- `top_percent` : percentage of brain voxels to average in the second plot (default `1.0`).
- `normalize`   : how to normalize time series for display.
                  `"demean"` (default) subtracts each series' temporal mean.
                  `"zscore"` standardises to zero mean, unit variance.
                  `"psc"` converts to percent signal change (100 × (y-mean)/mean).
                  `"none"` shows raw intensities.
- `peak_source` : `:first` (default) uses the first recon's peak/top-set for all recons;
                  `:per_recon` selects each recon's own peak/top-set independently.
- `save_dir`    : directory to save PNG files (default `nothing` = don't save).
- `save_name`   : base filename for saved figures (default `nothing`).
- `brain_mask`  : optional pre-computed brain mask (bypasses FSL `bet`).
- `design_matrix` : optional pre-built GLM design matrix.
- `show_model`  : overlay the fitted GLM model on each time series (default `true`).
- `show_task_shading` : shade task blocks in the background (default `true`).
- `show_spectrum` : show a power spectrum comparison figure (default `true`).
- `show_residuals` : show residual (data − model) time series for the top-n%
                     average (default `false`).
- `condition_names` : vector of condition names for task-shading legend
                      (default: `["Condition 1", ...]`).
- `time_range` : `(t_start, t_end)` in seconds to limit the displayed x-axis
                 (default `nothing` = show full scan).

# Returns
A named tuple `(peak_voxel_idx, top_voxel_indices)` with the brain-voxel
indices used, for downstream analysis.
"""
function compare_recons_time_series(
    schemes,
    recons,
    params::ExperimentParams;
    stat::String = "t",
    top_percent::Real = 1.0,
    normalize::String = "demean",
    peak_source::Symbol = :first,
    save_dir::Union{Nothing, AbstractString} = nothing,
    save_name::Union{Nothing, AbstractString} = nothing,
    brain_mask::Union{Nothing, BitArray{3}} = nothing,
    design_matrix::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    show_model::Bool = true,
    show_task_shading::Bool = true,
    show_spectrum::Bool = true,
    show_residuals::Bool = false,
    condition_names::Union{Nothing, Vector{String}} = nothing,
    time_range::Union{Nothing, Tuple{Real, Real}} = nothing)

    CairoMakie.update_theme!(fonts = (; regular = "TeX Gyre Heros"))

    n_cond = length(params.onsets)
    cond_names = isnothing(condition_names) ?
        ["Condition $i" for i in 1:n_cond] : condition_names

    ref_peak_idx = nothing
    ref_top_idx  = nothing

    for (scheme_base, scheme_label, scheme_prefix) in schemes
        n_recons = length(recons)

        # Reset per-scheme: brain-voxel indices are only valid within one mask
        ref_peak_idx = nothing
        ref_top_idx  = nothing

        peak_ts_raw      = Vector{Vector{Float32}}(undef, n_recons)
        peak_ts_model    = Vector{Vector{Float64}}(undef, n_recons)
        topn_ts_raw      = Vector{Vector{Float32}}(undef, n_recons)
        topn_ts_sem      = Vector{Vector{Float32}}(undef, n_recons)
        topn_ts_model    = Vector{Vector{Float64}}(undef, n_recons)
        peak_stat_vals   = Vector{Float64}(undef, n_recons)
        peak_r2_vals     = Vector{Float64}(undef, n_recons)
        topn_counts      = Vector{Int}(undef, n_recons)
        topn_resid_std   = Vector{Float64}(undef, n_recons)
        recon_labels     = Vector{String}(undef, n_recons)

        shared_mask  = brain_mask
        shared_dm    = design_matrix
        nt = 0

        for (ci, recon) in enumerate(recons)
            rtype, base, id = recon[1], recon[2], recon[3]
            recon_labels[ci] = recon[4]

            if rtype == :basic
                X = matread(joinpath(base, "$(scheme_base)_$(id).mat"))["img"]
            elseif rtype == :mslr
                vars    = matread(joinpath(base, id, "$(scheme_base).mat"))
                scale_n = length(recon) >= 5 ? recon[5] : nothing
                X = isnothing(scale_n) ?
                    dropdims(sum(vars["X"], dims=5), dims=5) :
                    vars["X"][:, :, :, :, scale_n]
            else
                error("Unknown recon type :$rtype — expected :basic or :mslr")
            end

            Y = Float32.(eltype(X) <: Complex ? abs.(X) : X)
            Y = Y[:, :, :, (params.n_discard+1):end]
            (nx, ny, nz, nt_local) = size(Y)
            nt = nt_local

            if isnothing(shared_mask)
                mean_vol    = dropdims(mean(Y, dims=4), dims=4)
                shared_mask = bet_brain_mask(mean_vol)
            end
            if isnothing(shared_dm)
                shared_dm = build_design_matrix(params.onsets, params.durations, nt, params.tr)
            end

            mask_flat = vec(shared_mask)
            Y_brain = Matrix{Float32}(transpose(reshape(Y, :, nt)))[:, mask_flat]
            t_brain, beta, _, z_brain, _ = run_glm(
                Y_brain, params.onsets, params.durations,
                params.contrast, nt, params.tr; design_matrix=shared_dm)

            scores = stat == "z" ? Float64.(z_brain) : Float64.(t_brain)

            # ── Peak voxel ────────────────────────────────────────────────────
            if peak_source == :first && !isnothing(ref_peak_idx)
                pk_idx = ref_peak_idx
            else
                pk_idx = argmax(scores)
                ci == 1 && (ref_peak_idx = pk_idx)
            end

            data_peak = Y_brain[:, pk_idx]
            model_peak = shared_dm * beta[:, pk_idx]
            peak_ts_raw[ci]    = data_peak
            peak_stat_vals[ci] = scores[pk_idx]
            peak_r2_vals[ci]   = _r_squared(Float64.(data_peak), model_peak)

            # ── Top n% voxels (positive scores only) ──────────────────────────
            if peak_source == :first && !isnothing(ref_top_idx)
                top_idx = ref_top_idx
            else
                thr = quantile(scores, 1.0 - top_percent / 100.0)
                top_idx = findall((scores .>= thr) .& (scores .> 0))
                isempty(top_idx) && (top_idx = [argmax(scores)])
                ci == 1 && (ref_top_idx = top_idx)
            end
            topn_counts[ci] = length(top_idx)
            Y_top = Y_brain[:, top_idx]
            topn_ts_raw[ci] = vec(mean(Y_top, dims=2))
            topn_ts_sem[ci] = length(top_idx) > 1 ?
                vec(std(Y_top, dims=2) ./ sqrt(length(top_idx))) :
                zeros(Float32, nt)

            # ── Fitted model for overlay / residuals ──────────────────────────
            peak_ts_model[ci] = model_peak
            topn_model = vec(mean(shared_dm * beta[:, top_idx], dims=2))
            topn_ts_model[ci] = topn_model

            # Residual std of averaged top-n% series (noise quality metric)
            topn_resid_std[ci] = std(Float64.(topn_ts_raw[ci]) .- topn_model)
        end

        # ── Print summary table ───────────────────────────────────────────────
        _print_ts_summary(scheme_label, recon_labels, peak_stat_vals,
                          peak_r2_vals, topn_counts, topn_resid_std, stat, top_percent)

        # ── Normalize time series ─────────────────────────────────────────────
        time_axis = collect(0:nt-1) .* Float64(params.tr)

        norm_peak   = [_normalize_ts(ts, normalize) for ts in peak_ts_raw]
        norm_topn   = [_normalize_ts(ts, normalize) for ts in topn_ts_raw]
        norm_sem    = Vector{Vector{Float64}}(undef, n_recons)
        norm_peak_m = Vector{Vector{Float64}}(undef, n_recons)
        norm_topn_m = Vector{Vector{Float64}}(undef, n_recons)

        for ci in 1:n_recons
            sf = _normalize_scale_factor(topn_ts_raw[ci], normalize)
            norm_sem[ci] = Float64.(topn_ts_sem[ci]) .* sf

            norm_peak_m[ci] = _normalize_ts_with_ref(
                Float64.(peak_ts_model[ci]), peak_ts_raw[ci], normalize)
            norm_topn_m[ci] = _normalize_ts_with_ref(
                Float64.(topn_ts_model[ci]), topn_ts_raw[ci], normalize)
        end

        # ── Y-axis label ──────────────────────────────────────────────────────
        ylabel = normalize == "demean"  ? "Signal (demeaned)" :
                 normalize == "zscore"  ? "Signal (z-scored)" :
                 normalize == "psc"     ? "% signal change" :
                                          "Signal (a.u.)"

        colors = _recon_colors(n_recons)

        # ── Figure 1: Peak voxel time series ──────────────────────────────────
        fig1 = _plot_time_series_figure(
            time_axis, norm_peak, recon_labels, colors, params, cond_names,
            "(a) Time course of most active voxel — $scheme_label", ylabel;
            show_task_shading = show_task_shading,
            time_range = time_range)
        display(fig1)

        if !isnothing(save_dir) && !isnothing(save_name)
            mkpath(save_dir)
            CairoMakie.save(joinpath(save_dir, "$(scheme_prefix)_$(save_name)_peak.png"), fig1; px_per_unit=2)
        end

        # ── Figure 2: Top n% averaged time series ────────────────────────────
        fig2 = _plot_time_series_figure(
            time_axis, norm_topn, recon_labels, colors, params, cond_names,
            "(b) Average time course of top $(top_percent)% voxels — $scheme_label", ylabel;
            show_task_shading = show_task_shading,
            time_range = time_range)
        display(fig2)

        if !isnothing(save_dir) && !isnothing(save_name)
            CairoMakie.save(joinpath(save_dir, "$(scheme_prefix)_$(save_name)_topn.png"), fig2; px_per_unit=2)
        end

        # ── Figure 3: Power spectrum of top-n% averaged time series ──────────
        if show_spectrum
            fig3 = _plot_spectrum_figure(
                topn_ts_raw, recon_labels, colors, params,
                "(c) Power spectra of top $(top_percent)% voxels — $scheme_label")
            display(fig3)

            if !isnothing(save_dir) && !isnothing(save_name)
                CairoMakie.save(joinpath(save_dir, "$(scheme_prefix)_$(save_name)_spectrum.png"), fig3; px_per_unit=2)
            end
        end

        # ── Figure 4: Residual time series (data − model) ────────────────────
        if show_residuals
            residuals = [norm_topn[ci] .- norm_topn_m[ci] for ci in 1:n_recons]
            resid_labels = [@sprintf("%s (σ=%.3f)", recon_labels[ci],
                            std(residuals[ci])) for ci in 1:n_recons]
            fig4 = _plot_time_series_figure(
                time_axis, residuals, resid_labels, colors, params, cond_names,
                "(d) Residuals (top $(top_percent)% average) — $scheme_label",
                "Residual ($ylabel)";
                show_task_shading = show_task_shading,
                time_range = time_range)
            CairoMakie.hlines!(CairoMakie.current_axis(), [0.0];
                color = :gray50, linewidth = 1.0, linestyle = :dash)
            display(fig4)

            if !isnothing(save_dir) && !isnothing(save_name)
                CairoMakie.save(joinpath(save_dir, "$(scheme_prefix)_$(save_name)_residuals.png"), fig4; px_per_unit=2)
            end
        end
    end

    return (peak_voxel_idx = ref_peak_idx, top_voxel_indices = ref_top_idx)
end


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers (module-internal, not exported)
# ─────────────────────────────────────────────────────────────────────────────

function _r_squared(y::AbstractVector{Float64}, yhat::AbstractVector{Float64})
    ss_res = sum((y .- yhat) .^ 2)
    ss_tot = sum((y .- mean(y)) .^ 2)
    return ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0
end

function _normalize_ts(ts::AbstractVector{<:Real}, mode::String)
    ts64 = Float64.(ts)
    m = mean(ts64)
    if mode == "demean"
        return ts64 .- m
    elseif mode == "zscore"
        s = std(ts64)
        return s > 0 ? (ts64 .- m) ./ s : ts64 .- m
    elseif mode == "psc"
        return abs(m) > eps() ? 100.0 .* (ts64 .- m) ./ m : ts64 .- m
    else
        return ts64
    end
end

function _normalize_ts_with_ref(ts::AbstractVector{<:Real},
                                 ref::AbstractVector{<:Real}, mode::String)
    ts64  = Float64.(ts)
    ref64 = Float64.(ref)
    m = mean(ref64)
    if mode == "demean"
        return ts64 .- m
    elseif mode == "zscore"
        s = std(ref64)
        return s > 0 ? (ts64 .- m) ./ s : ts64 .- m
    elseif mode == "psc"
        return abs(m) > eps() ? 100.0 .* (ts64 .- m) ./ m : ts64 .- m
    else
        return ts64
    end
end

function _normalize_scale_factor(ts::AbstractVector{<:Real}, mode::String)
    ts64 = Float64.(ts)
    m = mean(ts64)
    if mode == "demean"
        return 1.0
    elseif mode == "zscore"
        s = std(ts64)
        return s > 0 ? 1.0 / s : 1.0
    elseif mode == "psc"
        return abs(m) > eps() ? 100.0 / m : 1.0
    else
        return 1.0
    end
end

function _recon_colors(n::Int)
    base = [
        CairoMakie.RGBf(0.122, 0.467, 0.706),  # blue
        CairoMakie.RGBf(1.000, 0.498, 0.055),  # orange
        CairoMakie.RGBf(0.173, 0.627, 0.173),  # green
        CairoMakie.RGBf(0.839, 0.153, 0.157),  # red
        CairoMakie.RGBf(0.580, 0.404, 0.741),  # purple
        CairoMakie.RGBf(0.549, 0.337, 0.294),  # brown
        CairoMakie.RGBf(0.890, 0.467, 0.761),  # pink
        CairoMakie.RGBf(0.498, 0.498, 0.498),  # gray
    ]
    return [base[mod1(i, length(base))] for i in 1:n]
end

const _CONDITION_SHADING_COLORS = [
    (:steelblue, 0.3),
    (:salmon, 0.3),
    (:mediumseagreen, 0.3),
    (:mediumpurple, 0.3),
]

function _plot_time_series_figure(
    time_axis::AbstractVector, series::Vector, labels::Vector{String},
    colors::Vector, params::ExperimentParams,
    cond_names::Vector{String},
    title::String, ylabel::String;
    sem_bands::Union{Nothing, Vector} = nothing,
    show_task_shading::Bool = true,
    time_range::Union{Nothing, Tuple{Real, Real}} = nothing)

    n = length(series)
    fig = CairoMakie.Figure(size = (1200, 500), backgroundcolor = :white)

    x_lo = isnothing(time_range) ? 0.0 : Float64(time_range[1])
    x_hi = isnothing(time_range) ? time_axis[end] : Float64(time_range[2])
    x_ticks = collect(x_lo:20.0:x_hi)

    ax = CairoMakie.Axis(fig[1, 1];
        xlabel    = "Time (s)",
        ylabel    = ylabel,
        title     = title,
        titlesize = 18,
        xticks    = x_ticks,
        limits    = ((x_lo, x_hi), nothing))

    # ── Task-block shading (per-condition colors) ─────────────────────────
    if show_task_shading && !isempty(params.onsets)
        for (ci, (cond_onsets, cond_durs)) in enumerate(zip(params.onsets, params.durations))
            shade_color = _CONDITION_SHADING_COLORS[mod1(ci, length(_CONDITION_SHADING_COLORS))]
            first_block = true
            for (onset, dur) in zip(cond_onsets, cond_durs)
                t_end = onset + dur
                if t_end > x_lo && onset < x_hi
                    CairoMakie.vspan!(ax, Float64(onset), Float64(t_end);
                        color = shade_color,
                        label = first_block ? cond_names[ci] : nothing)
                    first_block = false
                end
            end
        end
    end

    # ── Data lines ────────────────────────────────────────────────────────
    for i in 1:n
        if !isnothing(sem_bands)
            upper = series[i] .+ sem_bands[i]
            lower = series[i] .- sem_bands[i]
            CairoMakie.band!(ax, time_axis, lower, upper;
                color = (colors[i], 0.2))
        end

        CairoMakie.lines!(ax, time_axis, series[i];
            color = colors[i], linewidth = 2.0, label = labels[i])
    end

    CairoMakie.axislegend(ax; position = :ct, framevisible = true,
        labelsize = 11, backgroundcolor = (:white, 0.85),
        orientation = :horizontal, nbanks = 1)

    return fig
end

function _plot_spectrum_figure(
    series::Vector{Vector{Float32}}, labels::Vector{String},
    colors::Vector, params::ExperimentParams, title::String)

    n = length(series)
    nt = length(series[1])
    fs = 1.0 / Float64(params.tr)
    f_max = 0.2

    # Task fundamental frequency from the block period
    all_onsets = sort(vcat(params.onsets...))
    task_freq = nothing
    if length(all_onsets) >= 2
        diffs = diff(Float64.(all_onsets))
        block_period = 2.0 * median(diffs)
        task_freq = 1.0 / block_period
    end

    # Build x-axis ticks: regular grid + task frequency labeled
    base_ticks = collect(0.0:0.025:f_max)
    tick_vals = copy(base_ticks)
    tick_labels = [@sprintf("%.3f", f) for f in base_ticks]
    if !isnothing(task_freq) && task_freq <= f_max
        # Replace the nearest base tick with the task freq label
        _, idx = findmin(abs.(base_ticks .- task_freq))
        tick_vals[idx] = task_freq
        tick_labels[idx] = @sprintf("%.4f\ntask", task_freq)
    end

    fig = CairoMakie.Figure(size = (1200, 500), backgroundcolor = :white)
    ax = CairoMakie.Axis(fig[1, 1];
        xlabel = "Frequency (Hz)",
        ylabel = "Power (dB)",
        title  = title,
        titlesize = 18,
        xticks = (tick_vals, tick_labels),
        xticklabelrotation = π/4,
        limits = ((0.0, f_max), nothing))

    for i in 1:n
        ts = Float64.(series[i]) .- mean(series[i])
        N = length(ts)
        freqs = (0:N÷2) .* (fs / N)
        spectrum = abs.(fft(ts))[1:N÷2+1]
        spectrum[2:end-1] .*= 2
        power_db = 10.0 .* log10.(spectrum .^ 2 .+ eps())

        CairoMakie.lines!(ax, freqs[2:end], power_db[2:end];
            color = colors[i], linewidth = 2.0, label = labels[i])
    end

    if !isnothing(task_freq) && task_freq <= f_max
        CairoMakie.vlines!(ax, [task_freq];
            color = :gray40, linewidth = 1.5, linestyle = :dash)
    end

    CairoMakie.axislegend(ax; position = :ct, framevisible = true,
        labelsize = 11, backgroundcolor = (:white, 0.85),
        orientation = :horizontal, nbanks = 1)

    return fig
end

function _print_ts_summary(scheme_label, recon_labels, peak_stat_vals,
                            peak_r2_vals, topn_counts, topn_resid_std,
                            stat, top_percent)
    println("\n── Time-series summary: $scheme_label")
    @printf("   %-35s  %6s  %6s  %6s  %8s\n",
            "Reconstruction", "peak $stat", "R²", "n_top", "σ_resid")
    println("   ", "─"^72)
    for ci in eachindex(recon_labels)
        @printf("   %-35s  %6.2f  %6.3f  %6d  %8.3f\n",
                recon_labels[ci], peak_stat_vals[ci], peak_r2_vals[ci],
                topn_counts[ci], topn_resid_std[ci])
    end
end
