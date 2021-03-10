module TestMiniBatch

using ParallelKMeans
using Test
using StableRNGs
using StatsBase
using Distances


@testset "MiniBatch default batch size" begin
    @test MiniBatch() == MiniBatch(100)
end


@testset "MiniBatch convergence" begin
    rng = StableRNG(2020)
    X = rand(rng, 3, 100)

    baseline = [kmeans(Lloyd(), X, 2).totalcost for i in 1:1_000] |> mean |> round
    # TODO: Switch to kmeans after full implementation
    res = [ParallelKMeans.kmeans!(MiniBatch(50), X, 2)[end] for i in 1:1_000] |> mean |> round

    @test baseline == res
end


@testset "MiniBatch metric support" begin
    rng = StableRNG(2020)
    X = rand(rng, 3, 100)

    baseline = [kmeans(Lloyd(), X, 2;
                       tol=1e-6, metric=Cityblock(),
                       max_iters=500).totalcost for i in 1:1000] |> mean |> floor
    # TODO: Switch to kmeans after full implementation
    res = [ParallelKMeans.kmeans!(MiniBatch(), X, 2;
                                  metric=Cityblock(), tol=1e-6,
                                  max_iters=500)[end] for i in 1:1000] |> mean |> floor

    @test baseline == res
end








end # module