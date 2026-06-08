# ─────────────────────────────────────────────────────────────────────────────
# Analysis Pipelines
# ─────────────────────────────────────────────────────────────────────────────
#
# High-level GLM analysis-and-plotting drivers. Included into the `FmriAnalysis`
# module by `src/fmri_analysis.jl`; relies on `Statistics` and the section 1–7
# functions imported/defined by the parent module, so it must not add its own
# `using` statements.

"""
    analyze_and_plot(X, params, title_base; ref_slice_idx=nothing,
                     brain_mask=nothing, design_matrix=nothing, tmp_dir="/tmp")

Run the full GLM pipeline on a single 4-D volume and display an orthogonal
slice plot. Returns the masked 4-D magnitude timeseries.
...
# Returns
- `slice_idx` : the NamedTuple slice index actually used.
- `t_vol`     : 3-D t-score volume `(nx, ny, nz)`.
- `Y_masked`  : 4-D masked magnitude timeseries `(nx, ny, nz, nt)`.
"""
function analyze_and_plot(X::AbstractArray{<:Number,4}, params::ExperimentParams,
    title_base::String; ref_slice_idx=nothing,
    brain_mask=nothing, design_matrix=nothing, tmp_dir::String="/tmp")

    # Discard instructional frames
    Y = X[:, :, :, (params.n_discard+1):end]
    (nx, ny, nz, nt) = size(Y)

    # Auto-convert complex input to magnitude
    if eltype(Y) <: Complex
        @warn "analyze_and_plot: complex input detected for \"$title_base\" — " *
              "applying abs.() before GLM fitting."
        Y = abs.(Y)
    end

    # ── Brain mask ──────────────────────────────────────────────────────────
    if isnothing(brain_mask)
        mean_vol = dropdims(mean(Float32.(Y), dims=4), dims=4)
        brain_mask = bet_brain_mask(mean_vol; tmp_dir=tmp_dir)
    end
    brain_mask_flat = vec(brain_mask)

    # ── Mask the 4-D Timeseries ─────────────────────────────────────────────
    Y_masked = Float32.(Y) .* brain_mask

    # GLM on brain voxels only
    Y_mat = Matrix{Float32}(transpose(reshape(Float32.(Y), :, nt)))
    Y_mat_brain = Y_mat[:, brain_mask_flat]

    t_map_brain, _, dm = run_glm(Y_mat_brain, params.onsets, params.durations,
        params.contrast, nt, params.tr; design_matrix=design_matrix)

    # Reconstruct full t_map with zeros outside the brain
    t_map = zeros(Float32, nx * ny * nz)
    t_map[brain_mask_flat] .= t_map_brain

    # ── Display threshold: top 1% of brain t-scores ─────────────────────────
    df = nt - size(dm, 2)
    display_threshold = quantile(abs.(t_map_brain), 0.99)
    display_threshold = max(display_threshold, eps(Float32))

    # Visualize
    tmap_summary(t_map_brain; title="t-map summary for $title_base")

    # fig_flat = plot_tmap_flat(t_map_brain; threshold=display_threshold, title="t-scores for $title_base")
    # display(fig_flat)

    t_vol = reshape(t_map, nx, ny, nz)

    underlay = dropdims(mean(Y_masked, dims=4), dims=4)
    underlay_range = (minimum(underlay), maximum(underlay))

    if isnothing(ref_slice_idx)
        peak_idx = argmax(abs.(t_vol))
        slice_idx = (x=peak_idx[1], y=peak_idx[2], z=peak_idx[3])
    else
        slice_idx = ref_slice_idx
    end

    fig = plot_tmap_slices(
        t_vol;
        underlay=underlay,
        slice_indices=slice_idx,
        threshold=[-display_threshold, display_threshold],
        underlay_range=underlay_range,
        title="t-scores for $title_base, 99th percentile |t| > $(round(display_threshold, digits=2))")
    display(fig)

    return slice_idx, t_vol, Y_masked
end


"""
    analyze_and_plot(X::AbstractArray{<:Number,5}, params, Nscales, patch_sizes,
                     title_base; ...)

Run GLM on each signal component of a multi-scale low-rank (MSLR)
reconstruction...
...
# Returns
- `slice_idx` : the NamedTuple slice index used
- `t_vols`    : vector of per-scale t-score volumes `(nx, ny, nz)`
- `Y_vols`    : vector of per-scale masked 4-D magnitude timeseries
"""
function analyze_and_plot(
    X::AbstractArray{<:Number,5},
    params::ExperimentParams,
    Nscales::Int,
    patch_sizes,
    title_base::String;
    ref_slice_idx=nothing,
    brain_mask=nothing,
    tmp_dir::String="/tmp",
    threshold_quantile::Real=0.99,
    plot_summary::Bool=false,
    plot_sum::Bool=false)

    # Auto-convert complex input to magnitude
    if eltype(X) <: Complex
        @warn "analyze_and_plot: complex input detected for \"$title_base\" — " *
              "applying abs.() before GLM fitting."
        X = abs.(X)
    end

    (nx, ny, nz, nt_raw, _) = size(X)
    @assert size(X, 5) == Nscales "Nscales=$Nscales does not match size(X,5)=$(size(X,5))"
    nt = nt_raw - params.n_discard

    # ── Brain mask: derived from the temporal mean of the summed reconstruction
    if isnothing(brain_mask)
        Y_sum_for_mask = dropdims(sum(X[:, :, :, (params.n_discard+1):end, :], dims=5), dims=5)
        mean_vol = dropdims(mean(Float32.(Y_sum_for_mask), dims=4), dims=4)
        brain_mask = bet_brain_mask(mean_vol; tmp_dir=tmp_dir)
    end
    brain_mask_flat = vec(brain_mask)
    df = nt - length(params.contrast)

    # ── Hoist design matrix: identical across all scales ────────────────────
    design_matrix = build_design_matrix(params.onsets, params.durations, nt, params.tr)

    # ── Pass 1: compute all t-maps and collect global statistics ───────────
    t_maps = Vector{Vector{Float32}}(undef, Nscales)
    Y_vols = Vector{Array{Float32,4}}(undef, Nscales)
    underlays = Vector{Array{Float32,3}}(undef, Nscales)

    for scale in 1:Nscales
        GC.gc()
        Y_scale = X[:, :, :, (params.n_discard+1):end, scale]

        # Mask the 4-D Timeseries for this scale
        Y_masked = Float32.(Y_scale) .* brain_mask
        Y_vols[scale] = Y_masked

        Y_mat = Matrix{Float32}(transpose(reshape(Float32.(Y_scale), :, nt)))
        Y_mat_brain = Y_mat[:, brain_mask_flat]

        t_map_brain, _, _ = run_glm(Y_mat_brain, params.onsets, params.durations,
            params.contrast, nt, params.tr; design_matrix=design_matrix)

        # Reconstruct full t_map with zeros outside the brain
        t_map = zeros(Float32, nx * ny * nz)
        t_map[brain_mask_flat] .= t_map_brain
        t_maps[scale] = t_map

        underlays[scale] = dropdims(mean(Y_masked, dims=4), dims=4)
    end

    # ── Sum reconstruction: slice index and/or same-scale plot ─────────────
    t_sum_brain_vec = Float32[]
    t_sum_vol_out = nothing
    Y_sum_masked_out = nothing
    underlay_sum_out = nothing

    if plot_sum || isnothing(ref_slice_idx)
        GC.gc()
        Y_sum = dropdims(sum(X[:, :, :, (params.n_discard+1):end, :], dims=5), dims=5)
        Y_sum_mat_brain = Matrix{Float32}(transpose(reshape(Float32.(Y_sum), :, nt)))[:, brain_mask_flat]
        t_sum_brain_vec, _, _ = run_glm(Y_sum_mat_brain, params.onsets, params.durations,
            params.contrast, nt, params.tr; design_matrix=design_matrix)
        t_sum_flat = zeros(Float32, nx * ny * nz)
        t_sum_flat[brain_mask_flat] .= t_sum_brain_vec
        t_sum_vol_local = reshape(t_sum_flat, nx, ny, nz)

        if isnothing(ref_slice_idx)
            peak_idx = argmax(abs.(t_sum_vol_local))
            ref_slice_idx = (x=peak_idx[1], y=peak_idx[2], z=peak_idx[3])
        end

        if plot_sum
            t_sum_vol_out = t_sum_vol_local
            Y_sum_masked_out = Float32.(Y_sum) .* brain_mask
            underlay_sum_out = dropdims(mean(Y_sum_masked_out, dims=4), dims=4)
        end
        GC.gc()
    end

    # ── Shared t-score color scale: symmetric around global max |t| ───────
    global_max_t = maximum(maximum(abs.(tm)) for tm in t_maps)
    if plot_sum && !isnothing(t_sum_vol_out)
        global_max_t = max(global_max_t, maximum(abs.(t_sum_vol_out)))
    end
    shared_clim = (-global_max_t, global_max_t)

    # ── Shared underlay intensity range: global min/max across all scales ──
    u_global_min = minimum(minimum(u) for u in underlays)
    u_global_max = maximum(maximum(u) for u in underlays)
    if plot_sum && !isnothing(underlay_sum_out)
        u_global_min = min(u_global_min, minimum(underlay_sum_out))
        u_global_max = max(u_global_max, maximum(underlay_sum_out))
    end
    shared_underlay_range = (u_global_min, u_global_max)

    # ── Pass 2: plot sum (same scale), then per-scale ─────────────────────
    if plot_sum && !isnothing(t_sum_vol_out)
        GC.gc()
        sum_thr = quantile(abs.(t_sum_brain_vec), threshold_quantile)
        sum_thr = max(Float32(sum_thr), eps(Float32))
        pct = round(Int, threshold_quantile * 100)
        nscales_str = "$Nscales $(Nscales == 1 ? "scale" : "scales")"
        sum_title = Nscales > 1 ?
            "$title_base, $nscales_str, sum, $(pct)th percentile |t| > $(round(sum_thr, digits=2))" :
            "$title_base, $nscales_str, $(pct)th percentile |t| > $(round(sum_thr, digits=2))"
        plot_summary && tmap_summary(t_sum_brain_vec; title=sum_title)
        fig_sum = plot_tmap_slices(
            t_sum_vol_out;
            underlay=underlay_sum_out,
            slice_indices=ref_slice_idx,
            threshold=[-sum_thr, sum_thr],
            clim=shared_clim,
            underlay_range=shared_underlay_range,
            title=sum_title)
        display(fig_sum)
    end

    # Skip per-scale plots when Nscales==1 (sum and single scale are identical).
    for scale in (Nscales > 1 ? (1:Nscales) : ())
        GC.gc()
        t_map = t_maps[scale]
        underlay = underlays[scale]

        # Per-scale display threshold: top threshold_quantile of brain t-scores
        display_threshold = quantile(abs.(t_map[brain_mask_flat]), threshold_quantile)
        display_threshold = max(display_threshold, eps(Float32))

        pct = round(Int, threshold_quantile * 100)
        scale_title = "$title_base, $Nscales $(Nscales == 1 ? "scale" : "scales"), scale = $(patch_sizes[scale]), $(pct)th percentile |t| > $(round(display_threshold, digits=2))"

        plot_summary && tmap_summary(t_map[brain_mask_flat]; title=scale_title)

        # fig_flat = plot_tmap_flat(t_map[brain_mask_flat]; threshold=display_threshold, title="t-scores for $title_base")
        # display(fig_flat)

        t_vol = reshape(t_map, nx, ny, nz)

        fig = plot_tmap_slices(
            t_vol;
            underlay=underlay,
            slice_indices=ref_slice_idx,
            threshold=[-display_threshold, display_threshold],
            clim=shared_clim,
            underlay_range=shared_underlay_range,
            title=scale_title)

        display(fig)
    end

    t_vols = [reshape(tm, nx, ny, nz) for tm in t_maps]
    return ref_slice_idx, t_vols, Y_vols, t_sum_vol_out, Y_sum_masked_out
end
