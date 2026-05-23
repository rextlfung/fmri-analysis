using Test, Random, Statistics, Distributions

include(joinpath(@__DIR__, "..", "src", "fmri_analysis.jl"))
using .FmriAnalysis

@testset "FmriAnalysis" begin

    @testset "HRF" begin
        hrf = canonical_hrf(0.5)
        @test maximum(hrf) ≈ 1.0                       # normalized to unit peak
        peak_time = (argmax(hrf) - 1) * 0.5
        @test 4.0 ≤ peak_time ≤ 8.0                    # peak between 4–8 s
        @test minimum(hrf) < 0                          # undershoot present
    end

    @testset "Design matrix" begin
        onsets    = [collect(0.0:40.0:120.0), collect(20.0:40.0:120.0)]
        durations = [fill(20.0, 4), fill(20.0, 4)]
        n_scans = 200
        X = build_design_matrix(onsets, durations, n_scans, 0.8)
        @test size(X) == (200, 3)          # 2 conditions + intercept
        @test all(X[:, end] .== 1.0)       # intercept column is all-ones
        @test all(-0.1 .≤ X[:, 1] .≤ 1.1) # regressor values are bounded
    end

    @testset "GLM recovers known β" begin
        Random.seed!(42)
        n_scans, n_vox = 120, 30
        beta_true = [3.0, -1.5, 5.0]
        X_design = [randn(n_scans, 2) ones(n_scans)]
        Y = X_design * beta_true' .+ 0.05 .* randn(n_scans, n_vox)
        beta_fit, residuals, XtXinv = fit_glm(X_design, Y)
        @test all(isapprox.(beta_fit[:, 1], beta_true; atol=0.05))
        @test size(residuals) == (n_scans, n_vox)
        @test size(XtXinv) == (3, 3)
    end

    @testset "t-scores for planted effect" begin
        Random.seed!(7)
        n_scans = 150
        X_design = [collect(range(0.0, 1.0, n_scans)) ones(n_scans)]
        beta_true = [10.0, 2.0]
        Y = X_design * beta_true .+ 0.1 .* randn(n_scans)
        Y_mat = reshape(Y, n_scans, 1)
        beta_fit, residuals, XtXinv = fit_glm(X_design, Y_mat)
        contrast = [1.0, 0.0]
        t = compute_tscores(beta_fit, residuals, XtXinv, contrast)
        @test length(t) == 1
        @test t[1] > 10.0   # strong planted effect → high t-score
    end

    @testset "FDR under pure null" begin
        # BH FDR applied to i.i.d. t-scores from the null distribution.
        # The number of false rejections should be very small.
        Random.seed!(99)
        df = 100
        dist = TDist(df)
        t_null = rand(dist, 5000)
        _, mask, _, _ = fdr_correct(t_null, df; q=0.05)
        @test sum(mask) / length(mask) < 0.01
    end

    @testset "Bonferroni under pure null" begin
        Random.seed!(11)
        df = 100
        dist = TDist(df)
        t_null = rand(dist, 5000)
        _, mask, _, _ = bonferroni_correct(t_null, df; alpha=0.05)
        @test sum(mask) / length(mask) < 0.005
    end

    @testset "run_glm pipeline" begin
        Random.seed!(3)
        n_scans = 100
        onsets    = [collect(0.0:40.0:80.0), collect(20.0:40.0:80.0)]
        durations = [fill(20.0, 3), fill(20.0, 3)]
        contrast  = [1.0, -1.0, 0.0]
        tr = 0.8
        Y = randn(n_scans, 20)
        t_map, beta, X = run_glm(Y, onsets, durations, contrast, n_scans, tr)
        @test length(t_map) == 20
        @test size(X, 1) == n_scans
        @test size(X, 2) == 3   # 2 conditions + intercept

        # Passing a pre-built design matrix should give identical results
        t_map2, _, _ = run_glm(Y, onsets, durations, contrast, n_scans, tr;
                                design_matrix=X)
        @test t_map ≈ t_map2
    end

    @testset "t_to_p symmetry" begin
        df = 50
        t = [0.0, 1.96, -1.96, 3.0]
        p = t_to_p(t, df)
        @test p[1] ≈ 1.0 atol=0.01          # t=0 → p≈1
        @test isapprox(p[2], p[3]; atol=1e-10)  # symmetric
        @test p[4] < p[2]                    # higher t → lower p
    end

end
