"""
FmriTscores

Task-based fMRI GLM analysis and visualization.

Sections
────────
  1.  Haemodynamic Response Function
  2.  Design Matrix
  3.  GLM Fitting & Contrast t-scores
  4.  FDR / Bonferroni Correction
  5.  Brain Mask Extraction
  6.  Visualization
  7.  Experiment Parameters
  8.  Analysis Pipelines

System dependency: FSL (`bet` must be on PATH) for brain mask extraction.
"""
module FmriTscores

using Statistics
using LinearAlgebra
using FFTW
using NIfTI
using Printf
using Plots
using CairoMakie
using Distributions
using SpecialFunctions: gamma

export canonical_hrf, build_design_matrix, fit_glm, compute_tscores, run_glm,
       t_to_p, fdr_correct, bonferroni_correct, bet_brain_mask,
       ExperimentParams, plot_design_matrix, tmap_summary, plot_tmap_flat,
       plot_tmap_slices, analyze_and_plot, analyze_and_plot_mslr, export_niftis


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Haemodynamic Response Function
# ─────────────────────────────────────────────────────────────────────────────

"""
    canonical_hrf(tr; peak=6.0, undershoot=16.0, peak_disp=1.0,
                  undershoot_disp=1.0, ratio=0.167)

Double-gamma canonical HRF (SPM-style), sampled at `tr` seconds.
Returns a normalized vector covering 32 s.
"""
function canonical_hrf(tr::Real;
    peak::Real=6.0,
    undershoot::Real=16.0,
    peak_disp::Real=1.0,
    undershoot_disp::Real=1.0,
    ratio::Real=0.167)

    t = 0:tr:32.0
    gamma_pdf(t, a, b) = t^(a - 1) * exp(-t / b) / (b^a * gamma(a))

    h = [gamma_pdf(ti, peak / peak_disp, peak_disp) -
         ratio * gamma_pdf(ti, undershoot / undershoot_disp, undershoot_disp)
         for ti in t]

    return h ./ maximum(abs.(h))
end


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Design Matrix
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_design_matrix(onsets, durations, n_scans, tr)

Construct an (n_scans × n_conditions + 1) design matrix X.

- `onsets`    : vector of vectors, onset times in **seconds** per condition
- `durations` : vector of vectors, duration in seconds per condition
- `n_scans`   : total number of TR volumes
- `tr`        : repetition time in seconds

The last column is a column of ones (intercept / baseline).
"""
function build_design_matrix(
    onsets::Vector{<:AbstractVector{<:Real}},
    durations::Vector{<:AbstractVector{<:Real}},
    n_scans::Int,
    tr::Real)

    oversampling = 16
    dt = tr / oversampling
    hrf_fine = canonical_hrf(dt)   # HRF sampled at fine resolution, not zero-order held

    n_cond = length(onsets)
    X = zeros(n_scans, n_cond + 1)

    for c in 1:n_cond
        n_fine = n_scans * oversampling
        stimulus = zeros(n_fine)

        for (onset, dur) in zip(onsets[c], durations[c])
            i_start = max(1, round(Int, onset / dt) + 1)
            i_end = min(n_fine, round(Int, (onset + dur) / dt))
            stimulus[i_start:i_end] .= 1.0
        end

        convolved = _conv(stimulus, hrf_fine)[1:n_fine]
        X[:, c] = convolved[oversampling:oversampling:end][1:n_scans]
    end

    X[:, end] .= 1.0   # intercept
    return X
end

"""FFT-based 1-D convolution — O(n log n) replacement for the naive O(n²) version."""
function _conv(u::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    n = length(u) + length(v) - 1
    N = nextpow(2, n)          # pad to next power of 2 for efficient FFT
    U = fft([Float64.(u); zeros(N - length(u))])
    V = fft([Float64.(v); zeros(N - length(v))])
    return real(ifft(U .* V))[1:n]
end


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GLM Fitting & Contrast t-scores
# ─────────────────────────────────────────────────────────────────────────────

"""
    fit_glm(X, Y)

OLS fit of Y = Xβ + ε for data matrix Y (n_scans × n_voxels).

Returns `beta`, `residuals`, and `(X'X)⁻¹`.
"""
function fit_glm(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    XtXinv = inv(X' * X)
    beta = XtXinv * (X' * Y)
    residuals = Y - X * beta
    return beta, residuals, XtXinv
end


"""
    compute_tscores(beta, residuals, XtXinv, contrast)

Voxel-wise t-scores for a contrast vector c:

    t = c'β / sqrt( σ² · c'(X'X)⁻¹c )

where σ² = RSS / (n − p). NaN values in t are replaced with 0.
"""
function compute_tscores(
    beta::AbstractMatrix{<:Real},
    residuals::AbstractMatrix{<:Real},
    XtXinv::AbstractMatrix{<:Real},
    contrast::AbstractVector{<:Real})

    n, _ = size(residuals)
    p = size(beta, 1)
    df = n - p

    sigma2 = max.(vec(sum(residuals .^ 2; dims=1)) ./ df, eps())
    c_var = dot(contrast, XtXinv * contrast)
    c_beta = contrast' * beta

    t = vec(c_beta) ./ sqrt.(sigma2 .* c_var)
    t[isnan.(t)] .= 0.0
    return t
end


"""
    run_glm(Y, onsets, durations, contrast, n_scans, tr; design_matrix=nothing)

Full pipeline: design matrix → GLM fit → t-scores.

# Arguments
- `Y`             : (n_scans × n_voxels) BOLD data matrix
- `onsets`        : onset times per condition in seconds
- `durations`     : durations per condition in seconds
- `contrast`      : contrast vector (length = n_conditions + 1, for the intercept)
- `n_scans`       : number of volumes
- `tr`            : repetition time in seconds
- `design_matrix` : optional pre-built design matrix; built from `onsets`/`durations`
                    when `nothing` (default). Pass a pre-built matrix to avoid
                    recomputing it across repeated calls with the same parameters.

# Returns `t_map`, `beta`, `X`
"""
function run_glm(
    Y::AbstractMatrix{<:Real},
    onsets::Vector{<:AbstractVector{<:Real}},
    durations::Vector{<:AbstractVector{<:Real}},
    contrast::AbstractVector{<:Real},
    n_scans::Int,
    tr::Real;
    design_matrix::Union{AbstractMatrix{<:Real},Nothing}=nothing)

    X = isnothing(design_matrix) ? build_design_matrix(onsets, durations, n_scans, tr) :
                                   design_matrix

    @assert size(Y, 1) == n_scans "Y rows must equal n_scans"
    @assert length(contrast) == size(X, 2) "Contrast length must equal n_regressors"

    beta, residuals, XtXinv = fit_glm(X, Y)
    t_map = compute_tscores(beta, residuals, XtXinv, contrast)

    return t_map, beta, X
end


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Multiple Comparisons Correction
# ─────────────────────────────────────────────────────────────────────────────

"""
    t_to_p(t, df; two_tailed=true)

Convert a vector of t-scores to p-values using `Distributions.TDist`.

- `df`         : residual degrees of freedom (n_scans - n_regressors)
- `two_tailed` : if true (default), returns two-tailed p-values
"""
function t_to_p(t::AbstractVector{<:Real}, df::Int; two_tailed::Bool=true)
    dist = TDist(df)
    p = ccdf.(dist, abs.(t))
    two_tailed && (p .*= 2)
    return clamp.(p, 0.0, 1.0)
end

"""
    bonferroni_correct(t_map, df; alpha=0.05, two_tailed=true)

Bonferroni correction for a voxel-wise t-map. Divides the desired alpha level
by the number of voxels to control the family-wise error rate (FWER) — i.e.
the probability of *any* false positive across the whole brain.

# Arguments
- `t_map`      : vector of t-scores (one per voxel)
- `df`         : residual degrees of freedom (n_scans - n_regressors)
- `alpha`      : desired FWER level (default 0.05)
- `two_tailed` : whether to use two-tailed p-values (default true)

# Returns
- `t_map_bonf` : t-map with sub-threshold voxels zeroed out
- `mask`       : BitVector — true for voxels surviving correction
- `p_vals`     : raw p-value for each voxel
- `t_threshold`: equivalent t-score cutoff

# Example
    t_map_bonf, mask, p_vals, t_thr = bonferroni_correct(t_map, df; alpha=0.05)
    println("Voxels surviving Bonferroni: ", sum(mask))
"""
function bonferroni_correct(t_map::AbstractVector{<:Real}, df::Int;
    alpha::Real=0.05, two_tailed::Bool=true)

    n = length(t_map)
    p = t_to_p(t_map, df; two_tailed)
    p_threshold = alpha / n        # Bonferroni-adjusted threshold

    mask = p .<= p_threshold
    t_threshold = any(mask) ? Float64(minimum(abs.(vec(t_map)[mask]))) : NaN
    t_map_bonf = t_map .* mask

    return t_map_bonf, mask, p, t_threshold
end

"""
    fdr_correct(t_map, df; q=0.05, two_tailed=true)

Benjamini-Hochberg FDR correction for a voxel-wise t-map.

# Arguments
- `t_map`      : vector of t-scores (one per voxel)
- `df`         : residual degrees of freedom (n_scans - n_regressors)
- `q`          : desired FDR level (default 0.05)
- `two_tailed` : whether to use two-tailed p-values (default true)

# Returns
- `t_map_fdr`  : t-map with sub-threshold voxels zeroed out
- `mask`       : BitVector — true for voxels surviving FDR correction
- `p_vals`     : raw p-value for each voxel
- `t_threshold`: the t-score cutoff corresponding to the FDR threshold
                 (NaN if no voxels survive)

# Example
    t_map_fdr, mask, p_vals, t_thr = fdr_correct(t_map, n_scans - size(X, 2))
    println("Voxels surviving FDR q<0.05: ", sum(mask))
"""
function fdr_correct(t_map::AbstractVector{<:Real}, df::Int;
    q::Real=0.05, two_tailed::Bool=true)

    n = length(t_map)
    p = t_to_p(t_map, df; two_tailed)

    # Benjamini-Hochberg: sort p-values, find largest k where p_(k) ≤ k/n · q
    sorted_p = sort(p)
    bh_line = (1:n) .* (q / n)
    surviving = sorted_p .<= bh_line

    mask = falses(n)
    if any(surviving)
        k_max = findlast(surviving)
        p_threshold = sorted_p[k_max]
        mask = p .<= p_threshold
        t_threshold = minimum(abs.(t_map[mask]))
    else
        t_threshold = NaN
    end

    t_map_fdr = t_map .* mask

    return t_map_fdr, mask, p, t_threshold
end


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Brain Mask Extraction
# ─────────────────────────────────────────────────────────────────────────────

"""
    bet_brain_mask(mean_vol; tmp_dir="/tmp")

Run FSL BET on a 3-D temporal mean volume and return a binary brain mask.

The volume is written to a temporary NIfTI file in `tmp_dir`, BET is invoked
as `bet <input> <output> -m -n` (mask output only, no brain-extracted volume),
the resulting mask is read back into Julia, and all temporary files are deleted.

# Arguments
- `mean_vol` : 3-D array (nx, ny, nz); the temporal mean of the BOLD series.
               Complex input is converted to magnitude automatically.
- `tmp_dir`  : directory for temporary NIfTI files (default: `"/tmp"`).

# Returns
- `mask` : `BitArray{3}` of size (nx, ny, nz); `true` inside the brain.

# Requirements
FSL must be installed and `bet` must be on `PATH`.
"""
function bet_brain_mask(mean_vol::AbstractArray{<:Number,3}; tmp_dir::String="/tmp")
    vol = eltype(mean_vol) <: Complex ? Float32.(abs.(mean_vol)) : Float32.(mean_vol)

    mkpath(tmp_dir)
    base      = joinpath(tmp_dir, "fmri_bet_$(getpid())")
    in_path   = base * "_input.nii"
    out_base  = base * "_output"
    mask_path = out_base * "_mask.nii.gz"

    try
        niwrite(in_path, NIVolume(vol))
        run(`bet $in_path $out_base -m -n`)
        return BitArray(Array(niread(mask_path)) .> 0)
    finally
        isfile(in_path)   && rm(in_path)
        isfile(mask_path) && rm(mask_path)
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Visualization
# ─────────────────────────────────────────────────────────────────────────────
"""
    plot_design_matrix(X; condition_names=nothing)

Heatmap of the GLM design matrix — useful for sanity-checking your model.
"""
function plot_design_matrix(X::AbstractMatrix{<:Real};
    condition_names::Union{Vector{String},Nothing}=nothing)

    n_regressors = size(X, 2)
    labels = isnothing(condition_names) ?
             ["Cond $i" for i in 1:(n_regressors-1)] :
             condition_names
    push!(labels, "Intercept")

    return Plots.heatmap(X;
        color=:grays,
        xlabel="Regressor",
        ylabel="Scan (TR)",
        title="Design matrix",
        xticks=(1:n_regressors, labels),
        size=(600, 400))
end

"""
    tmap_summary(t_map; thresholds=[1.65, 1.96, 2.58, 3.29, 4.42, 5.0], title=nothing)

Print a table of how many voxels survive common t-thresholds, with
approximate two-tailed p-values and percentage of total voxels.

Default thresholds correspond roughly to:
  p<.10, p<.05, p<.02, p<.01, p<.002, p<.001, p<.00001, p<.000001 (uncorrected)
"""
function tmap_summary(t_map::AbstractArray{<:Real};
    thresholds::Vector{Float64}=[1.65, 1.96, 2.33, 2.58, 3.09, 3.29, 4.42, 5.0],
    title::Union{String,Nothing}=nothing)

    total = length(t_map)
    header = isnothing(title) ? "── t-map summary" : "── t-map summary: $title"
    println("\n$header")
    @printf("   Total voxels : %d\n", total)
    @printf("   Mean t       : %+.3f\n", mean(t_map))
    @printf("   Std  t       : %.3f\n", std(t_map))
    @printf("   Min / Max    : %.3f  /  %.3f\n", minimum(t_map), maximum(t_map))
    @printf("   Median |t|   : %.3f\n", median(abs.(t_map)))
    @printf("   99th pct |t| : %.3f\n", quantile(abs.(t_map), 0.99))
    println("   ┌───────────┬────────────┬────────┬────────┬─────────┬────────┐")
    println("   │ threshold │  approx p  │  pos   │  neg   │  total  │   %    │")
    println("   ├───────────┼────────────┼────────┼────────┼─────────┼────────┤")
    approx_p = [0.10, 0.05, 0.02, 0.01, 0.002, 0.001, 0.00001, 0.000001]
    for (thr, p) in zip(thresholds, approx_p)
        pos = count(t_map .> thr)
        neg = count(t_map .< -thr)
        both = pos + neg
        pct = 100.0 * both / total
        @printf("   │   |t|>%-4.2f│  p<%-6.0e  │ %6d │ %6d │ %7d │ %5.1f%% │\n",
            thr, p, pos, neg, both, pct)
    end
    println("   └───────────┴────────────┴────────┴────────┴─────────┴────────┘")
end

"""
    plot_tmap_flat(t_map; threshold=2.0, title="t-score map")

Two-panel Plots.jl figure for a 1-D t-map vector:
  - Left  : bar chart colored by sign / threshold
  - Right : histogram with threshold lines
"""
function plot_tmap_flat(t_map::AbstractVector{<:Real};
    threshold=nothing,
    title::String="t-score map")

    threshold = isnothing(threshold) ? 1.96 : Float64(threshold)

    n = length(t_map)
    colors = [t >= threshold ? :crimson :
              t <= -threshold ? :dodgerblue : :lightgray
              for t in t_map]

    p1 = Plots.bar(1:n, t_map;
        color=colors,
        legend=false,
        xlabel="Voxel index",
        ylabel="t-score",
        title="per-voxel t-scores",
        linecolor=:match,
        ylims=(minimum(t_map) * 1.1, maximum(t_map) * 1.1))

    Plots.hline!(p1, [threshold, -threshold];
        linestyle=:dash, color=:black, linewidth=1.5, label="")

    p2 = Plots.histogram(t_map;
        bins=40,
        color=:steelblue,
        alpha=0.7,
        legend=false,
        xlabel="t-score",
        ylabel="Voxel count",
        title="t-score distribution")

    Plots.vline!(p2, [threshold, -threshold];
        linestyle=:dash, color=:black, linewidth=1.5, label="")

    return Plots.plot(p1, p2;
        layout=(1, 2),
        plot_title=title,
        size=(1000, 400),
        margin=5Plots.mm)
end

"""
    plot_tmap_slices(t_vol; threshold=[-1.96, 1.96], clim=nothing,
                     underlay=nothing, underlay_range=nothing,
                     title="t-map slices", slice_indices=nothing)

Orthogonal (axial / coronal / sagittal) slice view of a 3-D t-map using
CairoMakie. Sub-threshold voxels are transparent.

- `underlay`       : optional same-size anatomical volume shown in grayscale
- `underlay_range` : `(u_min, u_max)` for consistent anatomical scaling across
                     multiple calls; computed per-slice when omitted
- `slice_indices`  : NamedTuple `(x=i, y=j, z=k)`; defaults to peak |t| voxel

Returns a `CairoMakie.Figure` — call `display(fig)` or `save("out.png", fig)`.
"""
function plot_tmap_slices(t_vol::AbstractArray{<:Real,3};
    threshold=nothing,
    clim=nothing,
    underlay=nothing,
    underlay_range=nothing,
    title::String="t-map slices",
    slice_indices=nothing)

    t_vals = filter(x -> !isnan(x) && !iszero(x), vec(t_vol))

    threshold = isnothing(threshold) ? [-1.96f0, 1.96f0] : Float32.(threshold)
    clim = if isnothing(clim)
        isempty(t_vals) ? (-1.0f0, 1.0f0) : (minimum(t_vals), maximum(t_vals))
    else
        Float32.(clim)
    end

    sx, sy, sz = size(t_vol)
    if isnothing(slice_indices)
        abs_vol = abs.(t_vol)
        peak_idx = all(isnan, abs_vol) ? CartesianIndex(sx ÷ 2, sy ÷ 2, sz ÷ 2) :
                   argmax(replace(abs_vol, NaN => -Inf))
        si = (x=peak_idx[1], y=peak_idx[2], z=peak_idx[3])
    else
        si = slice_indices
    end

    masked = Float32.(t_vol)
    masked[masked.>threshold[1].&&masked.<threshold[2]] .= NaN32

    function get_slices(dim, idx)
        sl_t = Matrix(selectdim(masked, dim, idx))
        sl_u = isnothing(underlay) ? nothing : Matrix(selectdim(underlay, dim, idx))
        return sl_t, sl_u
    end

    slices = [
        ("Axial (z=$(si.z))",     get_slices(3, si.z)...),
        ("Coronal (y=$(si.y))",   get_slices(2, si.y)...),
        ("Sagittal (x=$(si.x))", get_slices(1, si.x)...),
    ]

    fig = CairoMakie.Figure(size=(2200, 840), backgroundcolor=:black)
    CairoMakie.Label(fig[0, 1:3],
        "$title, t ∉ [$(round(threshold[1], digits=2)), $(round(threshold[2], digits=2))]";
        fontsize=18, color=:white, font=:bold)

    sym_range = maximum(abs.(collect(clim)))

    for (col, (slab, sl_t, sl_u)) in enumerate(slices)
        ax = CairoMakie.Axis(fig[1, col];
            title=slab,
            titlecolor=:white,
            backgroundcolor=:black,
            aspect=CairoMakie.DataAspect(),
            yreversed=false,
            xticksvisible=false,
            yticksvisible=false,
            xticklabelsvisible=false,
            yticklabelsvisible=false)

        if !isnothing(sl_u)
            u_min, u_max = if !isnothing(underlay_range)
                Float32(underlay_range[1]), Float32(underlay_range[2])
            else
                minimum(sl_u), maximum(sl_u)
            end
            u_norm = (sl_u .- u_min) ./ (u_max - u_min + eps())
            CairoMakie.heatmap!(ax, u_norm; colormap=:grays, colorrange=(0, 1))
        end

        hm = CairoMakie.heatmap!(ax, sl_t;
            # Custom diverging colormap: Cyan -> Blue -> Black -> Red -> Yellow
            colormap=cgrad([:cyan, :blue, :black, :red, :yellow],
                           [0.0, 0.45, 0.5, 0.55, 1.0]),
            colorrange=(-sym_range, sym_range),
            nan_color=(:black, 0.0))

        col == 3 && CairoMakie.Colorbar(fig[1, 4], hm;
            label="t-score",
            labelcolor=:white,
            tickcolor=:white,
            ticklabelcolor=:white,
            width=16)
    end

    return fig
end

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Experiment Parameters
# ─────────────────────────────────────────────────────────────────────────────

"""
    ExperimentParams(; tr, onsets, durations, contrast, n_discard=12)

Experiment and GLM parameters, passed to `analyze_and_plot` and
`analyze_and_plot_mslr` to avoid hard-coding them in the analysis functions.

# Fields
- `tr`         : repetition time in seconds (`Float32`)
- `onsets`     : vector of onset-time vectors, one per condition (seconds)
- `durations`  : vector of duration vectors, one per condition (seconds)
- `contrast`   : contrast vector (length = n_conditions + 1, for the intercept)
- `n_discard`  : number of leading frames to discard (default: `12`)

# Example
    params = ExperimentParams(
        tr        = 0.8f0,
        onsets    = [collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
        durations = [fill(20.0f0, 9), fill(20.0f0, 9)],
        contrast  = [1.0f0, -1.0f0, 0.0f0])
"""
Base.@kwdef struct ExperimentParams
    tr::Float32
    onsets::Vector{Vector{Float32}}
    durations::Vector{Vector{Float32}}
    contrast::Vector{Float32}
    n_discard::Int = 12
end


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Analysis Pipelines
# ─────────────────────────────────────────────────────────────────────────────

"""
    analyze_and_plot(X, params, title_base; ref_slice_idx=nothing,
                     brain_mask=nothing, tmp_dir="/tmp")

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
    brain_mask=nothing, tmp_dir::String="/tmp")

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

    t_map_brain, _, design_matrix = run_glm(Y_mat_brain, params.onsets, params.durations,
        params.contrast, nt, params.tr)

    # Reconstruct full t_map with zeros outside the brain
    t_map = zeros(Float32, nx * ny * nz)
    t_map[brain_mask_flat] .= t_map_brain

    # ── FDR display threshold ───────────────────────────────────────────────
    df = nt - size(design_matrix, 2)
    _, _, _, t_thr = fdr_correct(t_map_brain, df)
    display_threshold = isnan(t_thr) ? quantile(abs.(t_map_brain), 0.99) : t_thr

    # Visualize
    tmap_summary(t_map_brain; title="t-map summary for $title_base")

    fig_flat = plot_tmap_flat(t_map_brain; title="t-scores for $title_base")
    display(fig_flat)

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
        title="t-scores for $title_base (FDR q<0.05)")
    display(fig)

    return slice_idx, t_vol, Y_masked
end


"""
    analyze_and_plot_mslr(...)

Run GLM on each signal component of a multi-scale low-rank (MSLR)
reconstruction...
...
# Returns
- `slice_idx` : the NamedTuple slice index used
- `t_vols`    : vector of per-scale t-score volumes `(nx, ny, nz)`
- `Y_vols`    : vector of per-scale masked 4-D magnitude timeseries
"""
function analyze_and_plot_mslr(
    X::AbstractArray{<:Number,5},
    params::ExperimentParams,
    Nscales::Int,
    patch_sizes,
    title_base::String;
    ref_slice_idx=nothing,
    brain_mask=nothing,
    tmp_dir::String="/tmp",
    q::Real=0.05,
    threshold_quantile::Real=0.99,
    plot_summary::Bool=false)

    # Auto-convert complex input to magnitude
    if eltype(X) <: Complex
        @warn "analyze_and_plot_mslr: complex input detected for \"$title_base\" — " *
              "applying abs.() before GLM fitting."
        X = abs.(X)
    end

    (nx, ny, nz, nt_raw, _) = size(X)
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
    fdr_thresholds = fill(NaN, Nscales)

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
        _, _, _, fdr_thresholds[scale] = fdr_correct(t_map_brain, df; q=q)
    end

    # ── Shared t-score color scale: symmetric around global max |t| ───────
    global_max_t = maximum(maximum(abs.(tm)) for tm in t_maps)
    shared_clim = (-global_max_t, global_max_t)

    # ── Shared underlay intensity range: global min/max across all scales ──
    u_global_min = minimum(minimum(u) for u in underlays)
    u_global_max = maximum(maximum(u) for u in underlays)
    shared_underlay_range = (u_global_min, u_global_max)

    # ── Determine slice index from summed reconstruction if not provided ───
    if isnothing(ref_slice_idx)
        Y_sum = dropdims(sum(X[:, :, :, (params.n_discard+1):end, :], dims=5), dims=5)
        Y_sum_mat = Matrix{Float32}(transpose(reshape(Float32.(Y_sum), :, nt)))
        Y_sum_mat_brain = Y_sum_mat[:, brain_mask_flat]

        t_sum_brain, _, _ = run_glm(Y_sum_mat_brain, params.onsets, params.durations,
            params.contrast, nt, params.tr; design_matrix=design_matrix)
        t_sum = zeros(Float32, nx * ny * nz)
        t_sum[brain_mask_flat] .= t_sum_brain

        t_sum_vol = reshape(t_sum, nx, ny, nz)
        peak_idx = argmax(abs.(t_sum_vol))
        ref_slice_idx = (x=peak_idx[1], y=peak_idx[2], z=peak_idx[3])
    end

    # ── Pass 2: plot every scale with per-scale FDR threshold ─────────────
    for scale in 1:Nscales
        GC.gc()
        scale_title = "$title_base, scale = $(patch_sizes[scale]) (FDR q<$q)"
        t_map = t_maps[scale]
        underlay = underlays[scale]

        plot_summary && tmap_summary(t_map[brain_mask_flat]; title=scale_title)

        # Per-scale display threshold
        scale_thr = fdr_thresholds[scale]
        display_threshold = if isnan(scale_thr)
            quantile(abs.(t_map[brain_mask_flat]), threshold_quantile)
        else
            scale_thr
        end

        fig_flat = plot_tmap_flat(t_map[brain_mask_flat]; title="t-scores for $title_base")
        display(fig_flat)

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
    return ref_slice_idx, t_vols, Y_vols
end

include("export.jl")

end # module FmriTscores
