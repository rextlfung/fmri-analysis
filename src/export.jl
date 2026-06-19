# ==============================================================================
# NIfTI export helper functions
# ==============================================================================

"""
    export_niftis(Y_masked, t_vol, prefix, out_dir; z_vol=nothing)

4-D method — Takes the pre-masked 4-D timeseries and 3-D t-map
returned by `analyze_and_plot`.

Writes (skipping any file that already exists):
  - `<prefix>_mag.nii`  : 4-D masked magnitude timeseries
  - `<prefix>_tmap.nii` : 3-D voxel-wise t-scores
  - `<prefix>_zmap.nii` : 3-D voxel-wise z-scores (when `z_vol` is provided)
"""
function export_niftis(Y_masked::AbstractArray{<:Real,4}, t_vol::AbstractArray{<:Real,3},
    prefix::String, out_dir::String; z_vol::Union{AbstractArray{<:Real,3},Nothing}=nothing)

    mag_path = joinpath(out_dir, "$(prefix)_mag.nii")
    if isfile(mag_path)
        @printf("Skipping %s — file already exists\n", mag_path)
    else
        niwrite(mag_path, NIVolume(Float32.(Y_masked)))
    end

    tmap_path = joinpath(out_dir, "$(prefix)_tmap.nii")
    if isfile(tmap_path)
        @printf("Skipping %s — file already exists\n", tmap_path)
    else
        niwrite(tmap_path, NIVolume(Float32.(t_vol)))
        @printf("Exported %s\n", prefix)
    end

    if !isnothing(z_vol)
        zmap_path = joinpath(out_dir, "$(prefix)_zmap.nii")
        if isfile(zmap_path)
            @printf("Skipping %s — file already exists\n", zmap_path)
        else
            niwrite(zmap_path, NIVolume(Float32.(z_vol)))
            @printf("Exported %s z-map\n", prefix)
        end
    end
end

"""
    export_niftis(Y_vols, t_vols, patch_sizes, Nscales, prefix, out_dir; z_vols=nothing)

5-D method — Takes the vectors of 4-D masked timeseries and 3-D t-maps
returned by the 5-D method of `analyze_and_plot`.

Writes per scale (skipping any file that already exists):
  - `<prefix>_<N>scales_patchsize<P>_mag.nii`  : 4-D masked magnitude timeseries
  - `<prefix>_<N>scales_patchsize<P>_tmap.nii` : 3-D voxel-wise t-scores
  - `<prefix>_<N>scales_patchsize<P>_zmap.nii` : 3-D voxel-wise z-scores (when `z_vols` is provided)
"""
function export_niftis(Y_vols::Vector{<:AbstractArray{<:Real,4}},
    t_vols::Vector{<:AbstractArray{<:Real,3}}, patch_sizes, Nscales::Int,
    prefix::String, out_dir::String;
    z_vols::Union{Vector{<:AbstractArray{<:Real,3}},Nothing}=nothing)

    for scale in 1:Nscales
        ps = Int.(patch_sizes[scale])
        tag = "$(prefix)_$(Nscales)scales_scale$(scale)_patchsize$(ps)"

        mag_path = joinpath(out_dir, "$(tag)_mag.nii")
        if isfile(mag_path)
            @printf("Skipping %s — file already exists\n", mag_path)
        else
            niwrite(mag_path, NIVolume(Float32.(Y_vols[scale])))
        end

        tmap_path = joinpath(out_dir, "$(tag)_tmap.nii")
        if isfile(tmap_path)
            @printf("Skipping %s — file already exists\n", tmap_path)
        else
            niwrite(tmap_path, NIVolume(Float32.(t_vols[scale])))
            @printf("Exported %s\n", tag)
        end

        if !isnothing(z_vols)
            zmap_path = joinpath(out_dir, "$(tag)_zmap.nii")
            if isfile(zmap_path)
                @printf("Skipping %s — file already exists\n", zmap_path)
            else
                niwrite(zmap_path, NIVolume(Float32.(z_vols[scale])))
                @printf("Exported %s z-map\n", tag)
            end
        end
    end
end
