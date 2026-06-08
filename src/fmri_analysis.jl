"""
FmriAnalysis

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

The high-level analysis pipelines (`analyze_and_plot`) live in
`scripts/run_analysis.jl`, included at the end of this file.

System dependency: FSL (`bet` must be on PATH) for brain mask extraction.
"""
module FmriAnalysis

using Statistics
using LinearAlgebra
using FFTW
using MAT
using NIfTI
using Printf
using Plots
using CairoMakie
using Distributions
using SpecialFunctions: gamma

export canonical_hrf, build_design_matrix, fit_glm, compute_tscores, run_glm,
       t_to_p, fdr_correct, bonferroni_correct, bet_brain_mask,
       ExperimentParams, plot_design_matrix, tmap_summary, plot_tmap_flat,
       plot_tmap_slices, analyze_and_plot, export_niftis, compare_recons


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
    hrf_fine = hrf_fine ./ sum(hrf_fine)   # unit-area: a sustained stimulus → amplitude ≈ 1
                                           # (a global scale on all condition columns; t-maps unchanged)

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
        X[:, c] = convolved[1:oversampling:end][1:n_scans]
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
        isfile(in_path)                  && rm(in_path)
        isfile(mask_path)                && rm(mask_path)
        isfile(out_base * ".nii.gz")     && rm(out_base * ".nii.gz")
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
             copy(condition_names)
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
        size=(1960, 900),
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
    max_t = isempty(t_vals) ? NaN : maximum(t_vals)

    threshold = isnothing(threshold) ? [-1.96f0, 1.96f0] : Float32.(threshold)
    clim = if isnothing(clim)
        isempty(t_vals) ? (-1.0f0, 1.0f0) : (minimum(t_vals), maximum(t_vals))
    else
        Float32.(clim)
    end

    sx, sy, sz = size(t_vol)

    # Auto-detect top-2 peaks by absolute t-score in the current volume
    abs_vol = abs.(t_vol)
    abs_vol_inf = replace(abs_vol, NaN => -Inf)
    all_nan = all(isnan, abs_vol)

    peak1_cart = all_nan ? CartesianIndex(sx ÷ 2, sy ÷ 2, sz ÷ 2) : argmax(abs_vol_inf)

    abs_vol_inf2 = copy(abs_vol_inf)
    abs_vol_inf2[peak1_cart] = -Inf
    peak2_cart = any(x -> x > -Inf, abs_vol_inf2) ?
        argmax(abs_vol_inf2) : CartesianIndex(sx ÷ 2, sy ÷ 2, sz ÷ 2 + 1)

    # Row 1: use provided slice_indices (ref alignment) or the detected peak
    si1 = isnothing(slice_indices) ?
        (x=peak1_cart[1], y=peak1_cart[2], z=peak1_cart[3]) : slice_indices
    # Row 2: always the literal 2nd-highest |t| voxel
    si2 = (x=peak2_cart[1], y=peak2_cart[2], z=peak2_cart[3])

    masked = Float32.(t_vol)
    masked[masked.>threshold[1].&&masked.<threshold[2]] .= NaN32

    function get_slices(si)
        function inner(dim, idx)
            sl_t = Matrix(selectdim(masked, dim, idx))
            sl_u = isnothing(underlay) ? nothing : Matrix(selectdim(underlay, dim, idx))
            return sl_t, sl_u
        end
        return [
            ("Axial (z=$(si.z))",     inner(3, si.z)...),
            ("Coronal (y=$(si.y))",   inner(2, si.y)...),
            ("Sagittal (x=$(si.x))", inner(1, si.x)...),
        ]
    end

    fig = CairoMakie.Figure(size=(1600, 1200), backgroundcolor=:black)
    CairoMakie.Label(fig[0, 1:3],
        "$title, max t = $(round(max_t, digits=2))";
        fontsize=18, color=:white, font=:bold)

    sym_range = maximum(abs.(collect(clim)))

    local colorbar_hm
    for (row, slices) in enumerate([get_slices(si1), get_slices(si2)])
        for (col, (slab, sl_t, sl_u)) in enumerate(slices)
            ax = CairoMakie.Axis(fig[row, col];
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

            row == 1 && col == 3 && (colorbar_hm = hm)
        end
    end

    CairoMakie.Colorbar(fig[1:2, 4], colorbar_hm;
        label="t-score",
        labelcolor=:white,
        tickcolor=:white,
        ticklabelcolor=:white,
        width=16)

    return fig
end

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Experiment Parameters
# ─────────────────────────────────────────────────────────────────────────────

"""
    ExperimentParams(; tr, onsets, durations, contrast, n_discard=12)

Experiment and GLM parameters, passed to `analyze_and_plot` to avoid
hard-coding them in the analysis functions.

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


include("../scripts/run_analysis.jl")
include("../scripts/compare_recons.jl")
include("export.jl")

end # module FmriAnalysis
