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

    baseline = [kmeans(Lloyd(), X, 2; max_iters=100_000).totalcost for i in 1:200] |> mean |> round

    res = [kmeans(MiniBatch(10), X, 2; max_iters=100_000).totalcost for i in 1:200] |> mean |> round

    @test baseline == res
end


@testset "MiniBatch metric support" begin
    rng = StableRNG(2020)
    X = rand(rng, 3, 100)

    baseline = [kmeans(Lloyd(), X, 2; metric=Cityblock(), max_iters=100_000).totalcost for i in 1:200] |> mean |> round

    res = [kmeans(MiniBatch(10), X, 2; metric=Cityblock(), max_iters=100_000).totalcost for i in 1:200] |> mean |> round

    @test baseline == res
end








end # module