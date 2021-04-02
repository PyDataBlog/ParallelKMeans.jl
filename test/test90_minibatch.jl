module TestMiniBatch

using ParallelKMeans
using Test
using StableRNGs
using StatsBase
using Distances
using Suppressor


@testset "MiniBatch default batch size" begin
    @test MiniBatch() == MiniBatch(100)
end


@testset "MiniBatch convergence" begin
    rng = StableRNG(2020)
    X = rand(rng, 3, 100)
    rng_orig = deepcopy(rng)

    # use lloyd as baseline
    baseline = [kmeans(Lloyd(), X, 2; max_iters=100_000).totalcost for i in 1:200] |> mean |> round
    # mini batch results
    res = [kmeans(MiniBatch(10), X, 2; max_iters=100_000).totalcost for i in 1:200] |> mean |> round
    # test for verbosity with convergence
    r = @capture_out kmeans(MiniBatch(50), X, 2; max_iters=2_000, verbose=true, rng=rng_orig)

    @test baseline == res
    @test endswith(r, "Successfully terminated with convergence.\n")
end

@testset "MiniBatch non-convergence warning" begin
    rng = StableRNG(2020)
    X = rand(rng, 3, 100)

    # Capture output and compare
    r = @capture_out kmeans(MiniBatch(10), X, 2; n_threads=1, max_iters=10, verbose=true, rng = rng)
    @test endswith(r, "Clustering model failed to converge. Labelling data with latest centroids.\n")
end

@testset "MiniBatch metric support" begin
    rng = StableRNG(2020)
    X = rand(rng, 3, 100)

    baseline = [kmeans(Lloyd(), X, 2; metric=Cityblock(), max_iters=100_000).totalcost for i in 1:200] |> mean |> round

    res = [kmeans(MiniBatch(10), X, 2; metric=Cityblock(), max_iters=100_000).totalcost for i in 1:200] |> mean |> round

    @test baseline == res
end








end # module