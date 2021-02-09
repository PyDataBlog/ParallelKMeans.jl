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
    X = [1 1 1 4 4 4 4 0 2 3 5 1; 2 4 0 2 0 4 5 1 2 2 5 -1.]

    rng = StableRNG(2020)
    baseline = kmeans(Lloyd(), X, 2, rng = rng)

    rng = StableRNG(2020)
    res = kmeans(MiniBatch(6), X, 2, rng = rng)

    @test baseline.totalcost ≈ res.totalcost
end


@testset "MiniBatch metric support" begin

end















end # module