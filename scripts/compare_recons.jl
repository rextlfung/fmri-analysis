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
    threshold_quantile::Real = 0.99f0,
    slice_indices::Union{Nothing, NamedTuple} = nothing,
    save_dir::Union{Nothing, AbstractString} = nothing,
    save_name::Union{Nothing, AbstractString} = nothing,
    stat::String = "t")

    CairoMakie.update_theme!(fonts = (; regular = "TeX Gyre Heros"))

    ref_si = slice_indices

    for (scheme_base, scheme_label, scheme_prefix) in schemes
        n_recons  = length(recons)
        t_vols    = Vector{Array{Float32,3}}(undef, n_recons)
        z_vols    = Vector{Array{Float64,3}}(undef, n_recons)
        underlays = Vector{Array{Float32,3}}(undef, n_recons)
        shared_mask = nothing
        shared_dm   = nothing
        vol_size    = nothing

        # ── Load data and run GLM for each reconstruction ──────────────────────
        for (ci, recon) in enumerate(recons)
            Y = _load_recon(recon, scheme_base, params.n_discard)
            (nx, ny, nz, nt) = size(Y)
            isnothing(vol_size) && (vol_size = (nx, ny, nz))

            if isnothing(shared_mask)
                shared_mask = bet_brain_mask(dropdims(mean(Y, dims=4), dims=4))
                shared_dm   = build_design_matrix(params.onsets, params.durations, nt, params.tr)
            end
            mask_flat = vec(shared_mask)

            glm = _brain_glm(Y, mask_flat, params; design_matrix=shared_dm)
            t_vols[ci] = _unflatten_to_volume(glm.t_brain, mask_flat, (nx, ny, nz))
            z_vols[ci] = _unflatten_to_volume(glm.z_brain, mask_flat, (nx, ny, nz); T=Float64)
            underlays[ci] = dropdims(mean(Y, dims=4), dims=4) .* shared_mask
        end

        # ── Slice indices: peak positive t in the first recon ──────────────────
        if isnothing(ref_si)
            peak_cart = argmax(t_vols[1])
            ref_si = (x=peak_cart[1], y=peak_cart[2], z=peak_cart[3])
        end
        si = ref_si

        # ── Select stat volumes to display ────────────────────────────────────────
        display_vols = stat == "z" ? z_vols : t_vols

        # ── Per-recon stats for annotation ────────────────────────────────────────
        pct99_vals = [Float32(quantile(abs.(vec(sv)[vec(shared_mask)]), Float64(threshold_quantile)))
                      for sv in display_vols]
        max_t_vals = [Float32(maximum(abs.(sv))) for sv in display_vols]

        # ── Shared colour scale across all recons ──────────────────────────────
        sym_range = maximum(max_t_vals)

        # ── Display threshold from first recon's brain voxels ──────────────────
        display_thr = max(pct99_vals[1], eps(Float32))

        # ── Build comparison figure ────────────────────────────────────────────
        CELL_W  = 300
        LABEL_W = 65
        CB_W    = 65
        HDR_H   = 75
        (vx, vy, vz) = vol_size
        h_ax  = round(Int, CELL_W * vy / vx)
        h_cor = round(Int, CELL_W * vz / vx)
        h_sag = round(Int, CELL_W * vz / vy)
        fig = CairoMakie.Figure(
            size            = (n_recons*CELL_W + LABEL_W + CB_W,
                               HDR_H + h_ax + h_cor + h_sag),
            figure_padding  = 0,
            backgroundcolor = :black,
        )

        # Layout: row 0 = headers, rows 1-3 = images
        #         col 1 = row labels, cols 2..n+1 = images, col n+2 = colorbar
        view_info = [
            ("Axial\n(z = $(si.z))",    3, si.z),
            ("Coronal\n(y = $(si.y))",  2, si.y),
            ("Sagittal\n(x = $(si.x))", 1, si.x),
        ]

        u_ranges = map(underlays) do ul
            vals = vcat([vec(Matrix(selectdim(ul, d, i))) for (_, d, i) in view_info]...)
            extrema(vals)
        end

        for (c, recon) in enumerate(recons)
            rlabel = recon[4]
            hdr = @sprintf("%s\n|%s| threshold = %.2f\nmax |%s| = %.2f",
                           rlabel, stat, pct99_vals[c], stat, max_t_vals[c])
            CairoMakie.Label(fig[0, c + 1], hdr;
                fontsize      = 19,
                color         = :white,
                halign        = :center,
                valign        = :bottom,
                justification = :center)
        end

        local last_hm
        for (r, (vname, dim, idx)) in enumerate(view_info)
            CairoMakie.Label(fig[r, 1], vname;
                fontsize = 18,
                color    = :white,
                halign   = :right,
                valign   = :center,
                rotation = π / 2)

            for (c, (stat_vol, underlay)) in enumerate(zip(display_vols, underlays))
                ax = CairoMakie.Axis(fig[r, c + 1];
                    backgroundcolor    = :black,
                    aspect             = CairoMakie.DataAspect(),
                    yreversed          = false,
                    xticksvisible      = false,
                    yticksvisible      = false,
                    xticklabelsvisible = false,
                    yticklabelsvisible = false)

                sl_u    = Matrix(selectdim(underlay, dim, idx))
                u_min, u_max = u_ranges[c]
                u_norm  = (sl_u .- u_min) ./ (u_max - u_min + eps(Float32))
                CairoMakie.heatmap!(ax, u_norm; colormap = :grays, colorrange = (0, 1))

                sl_t    = copy(Float32.(Matrix(selectdim(stat_vol, dim, idx))))
                hide    = (sl_t .> -display_thr) .& (sl_t .< display_thr)
                sl_t[hide] .= NaN32
                hm = CairoMakie.heatmap!(ax, sl_t;
                    colormap   = STAT_COLORMAP,
                    colorrange = (-sym_range, sym_range),
                    nan_color  = (:black, 0.0))

                r == length(view_info) && c == n_recons && (last_hm = hm)
            end
        end

        CairoMakie.Label(fig[0, n_recons + 2], stat;
            fontsize = 19,
            color    = :white,
            halign   = :center,
            valign   = :bottom)
        CairoMakie.Colorbar(fig[1:3, n_recons + 2], last_hm;
            labelvisible   = false,
            tickcolor      = :white,
            ticklabelcolor = :white,
            width          = 16)

        CairoMakie.colsize!(fig.layout, 1, CairoMakie.Fixed(LABEL_W))
        for c in 1:n_recons
            CairoMakie.colsize!(fig.layout, c + 1, CairoMakie.Fixed(CELL_W))
        end
        CairoMakie.colsize!(fig.layout, n_recons + 2, CairoMakie.Fixed(CB_W))
        CairoMakie.rowsize!(fig.layout, 0, CairoMakie.Fixed(HDR_H))
        CairoMakie.rowsize!(fig.layout, 1, CairoMakie.Fixed(h_ax))
        CairoMakie.rowsize!(fig.layout, 2, CairoMakie.Fixed(h_cor))
        CairoMakie.rowsize!(fig.layout, 3, CairoMakie.Fixed(h_sag))
        CairoMakie.colgap!(fig.layout, 0)
        CairoMakie.rowgap!(fig.layout, 0)

        display(fig)
        _maybe_save_figure(fig, save_dir,
            isnothing(save_name) ? nothing : "$(scheme_prefix)_$(save_name).png";
            px_per_unit=1)
    end

    return ref_si
end
