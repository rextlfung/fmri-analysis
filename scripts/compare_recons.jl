# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction Comparison Plot
# ─────────────────────────────────────────────────────────────────────────────
#
# Defines `compare_recons`, a high-level driver that loops over sampling schemes
# and produces one orthogonal-slice comparison figure per scheme, with one column
# per reconstruction method and three rows (axial / coronal / sagittal).
#
# Included into the `FmriAnalysis` module by `src/fmri_analysis.jl`; relies on
# `Statistics`, `CairoMakie`, `MAT`, and sections 1–7 of the parent module.
# Must not add its own `using` statements.

"""
    compare_recons(schemes, recons, params; threshold_quantile=0.99f0)

For each sampling scheme, run the GLM on every reconstruction and display a
single comparison figure.  The figure has one column per reconstruction and
three rows showing the axial, coronal, and sagittal slices that intersect at
the voxel with the highest positive t-score in the first reconstruction.
All columns are pinned to those same slice indices.

# Arguments
- `schemes` : vector of 3-tuples `(file_base, display_label, _)`.
              `file_base` is used to build per-recon `.mat` filenames.
              The third element (export prefix) is accepted but ignored, so an
              existing `schemes` vector can be passed verbatim.
- `recons`  : vector of tuples `(type, base_dir, identifier, display_label[, scale_n])`.
              `type` is `:basic` or `:mslr`:
              - `:basic` → `\$(base_dir)/\$(file_base)_\$(identifier).mat`, key `"img"` (4-D)
              - `:mslr`  → `\$(base_dir)/\$(identifier)/\$(file_base).mat`,  key `"X"`  (5-D).
                           Without `scale_n` (4-tuple): sums all scales.
                           With `scale_n::Int` (5-tuple): extracts the n-th scale (1-based).
- `params`  : `ExperimentParams` with scan timing and GLM settings.
- `threshold_quantile` : percentile of |t| in the first recon's brain voxels used
              to threshold the overlay (default `0.99`).

Each scheme produces one `CairoMakie.Figure` displayed via `display`.
"""
function compare_recons(
    schemes,
    recons,
    params::ExperimentParams;
    threshold_quantile::Real = 0.99f0)

    colormap_spec = cgrad([:cyan, :blue, :black, :red, :yellow],
                          [0.0, 0.45, 0.5, 0.55, 1.0])

    for (scheme_base, scheme_label, _) in schemes
        n_recons  = length(recons)
        t_vols    = Vector{Array{Float32,3}}(undef, n_recons)
        underlays = Vector{Array{Float32,3}}(undef, n_recons)
        shared_mask = nothing
        shared_dm   = nothing

        # ── Load data and run GLM for each reconstruction ──────────────────────
        for (ci, recon) in enumerate(recons)
            rtype, base, id = recon[1], recon[2], recon[3]
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
            (nx, ny, nz, nt) = size(Y)

            # Brain mask and design matrix are shared within a scheme
            if isnothing(shared_mask)
                mean_vol    = dropdims(mean(Y, dims=4), dims=4)
                shared_mask = bet_brain_mask(mean_vol)
                shared_dm   = build_design_matrix(params.onsets, params.durations, nt, params.tr)
            end
            mask_flat = vec(shared_mask)

            Y_brain = Matrix{Float32}(transpose(reshape(Y, :, nt)))[:, mask_flat]
            t_brain, _, _ = run_glm(Y_brain, params.onsets, params.durations,
                                    params.contrast, nt, params.tr; design_matrix=shared_dm)

            t_flat = zeros(Float32, nx * ny * nz)
            t_flat[mask_flat] .= t_brain
            t_vols[ci]    = reshape(t_flat, nx, ny, nz)
            underlays[ci] = dropdims(mean(Y .* Float32.(shared_mask), dims=4), dims=4)
        end

        # ── Slice indices: peak positive t in the first recon ──────────────────
        peak_cart = argmax(t_vols[1])
        si = (x=peak_cart[1], y=peak_cart[2], z=peak_cart[3])

        # ── Per-recon stats for annotation ────────────────────────────────────────
        pct99_vals = [Float32(quantile(abs.(t[shared_mask]), Float64(threshold_quantile)))
                      for t in t_vols]
        max_t_vals = [Float32(maximum(abs.(t))) for t in t_vols]

        # ── Shared colour scale across all recons ──────────────────────────────
        sym_range = maximum(max_t_vals)

        # ── Display threshold from first recon's brain voxels ──────────────────
        display_thr = max(pct99_vals[1], eps(Float32))

        # ── Build comparison figure ────────────────────────────────────────────
        fig = CairoMakie.Figure(
            size            = (320 * n_recons + 120, 980),
            backgroundcolor = :black)

        CairoMakie.Label(fig[0, 1:n_recons],
            scheme_label;
            fontsize  = 20,
            color     = :white,
            font      = :bold,
            tellwidth = false)

        for (col, recon) in enumerate(recons)
            rlabel = recon[4]
            hdr = @sprintf("%s\n99th |t| = %.2f  max |t| = %.2f",
                           rlabel, pct99_vals[col], max_t_vals[col])
            CairoMakie.Label(fig[1, col], hdr;
                fontsize  = 13,
                color     = :white,
                tellwidth = false)
        end

        view_info = [
            ("Axial\n(z = $(si.z))",    3, si.z),
            ("Coronal\n(y = $(si.y))",  2, si.y),
            ("Sagittal\n(x = $(si.x))", 1, si.x),
        ]

        local last_hm
        for (row, (vname, dim, idx)) in enumerate(view_info)
            CairoMakie.Label(fig[row + 1, 0], vname;
                fontsize   = 12,
                color      = :white,
                rotation   = π / 2,
                tellheight = false)

            for (col, (t_vol, underlay)) in enumerate(zip(t_vols, underlays))
                ax = CairoMakie.Axis(fig[row + 1, col];
                    backgroundcolor    = :black,
                    aspect             = CairoMakie.DataAspect(),
                    yreversed          = false,
                    xticksvisible      = false,
                    yticksvisible      = false,
                    xticklabelsvisible = false,
                    yticklabelsvisible = false)

                sl_u    = Matrix(selectdim(underlay, dim, idx))
                u_min, u_max = extrema(underlay)
                u_norm  = (sl_u .- u_min) ./ (u_max - u_min + eps(Float32))
                CairoMakie.heatmap!(ax, u_norm; colormap = :grays, colorrange = (0, 1))

                sl_t    = copy(Float32.(Matrix(selectdim(t_vol, dim, idx))))
                hide    = (sl_t .> -display_thr) .& (sl_t .< display_thr)
                sl_t[hide] .= NaN32
                hm = CairoMakie.heatmap!(ax, sl_t;
                    colormap   = colormap_spec,
                    colorrange = (-sym_range, sym_range),
                    nan_color  = (:black, 0.0))

                row == length(view_info) && col == n_recons && (last_hm = hm)
            end
        end

        CairoMakie.Colorbar(fig[2:4, n_recons + 1], last_hm;
            label          = "t-score",
            labelcolor     = :white,
            tickcolor      = :white,
            ticklabelcolor = :white,
            width          = 16)

        display(fig)
    end
end
