module TestLloyd

using ParallelKMeans
using Test
using StableRNGs
using StatsBase
using Distances


@testset "basic kmeans" begin
    X = [1. 2. 4.;]
    res = kmeans(X, 1; n_threads = 1, tol = 1e-6, verbose = false)
    @test res.assignments == [1, 1, 1]
    @test res.centers[1] ≈ 2.3333333333333335
    @test res.totalcost ≈ 4.666666666666666
    @test res.converged

    res = kmeans(X, 2; n_threads = 1, init = [1.0 4.0], tol = 1e-6, verbose = false)
    @test res.assignments == [1, 1, 2]
    @test res.centers ≈ [1.5 4.0]
    @test res.totalcost ≈ 0.5
    @test res.converged
end

@testset "no convergence yield last result" begin
    X = [1. 2. 4.;]
    res = kmeans(X, 2; n_threads = 1, init = [1.0 4.0], tol = 1e-6, max_iters = 1, verbose = false)
    @test !res.converged
    @test res.totalcost ≈ 0.5
end

@testset "singlethread linear separation" begin
    rng = StableRNG(2020)

    X = rand(rng, 3, 100)
    res = kmeans(X, 3; n_threads = 1, tol = 1e-6, verbose = false, rng = rng)

    @test res.totalcost ≈ 14.133433380466027
    @test res.converged
    @test res.iterations == 5
end

@testset "multithread linear separation quasi two threads" begin
    rng = StableRNG(2020)

    X = rand(rng, 3, 100)
    res = kmeans(X, 3; n_threads = 2, tol = 1e-6, verbose = false, rng = rng)

    @test res.totalcost ≈ 14.133433380466027
    @test res.converged
end

@testset "Lloyd Float32 support" begin
    rng = StableRNG(2020)
    X = Float32.(rand(rng, 3, 100))

    res = kmeans(Lloyd(), X, 3; n_threads = 1, tol = 1e-6, verbose = false, rng = rng)

    @test typeof(res.totalcost) == Float32
    @test res.totalcost ≈ 14.133433f0
    @test res.converged
    @test res.iterations == 5

    rng = StableRNG(2020)
    X = Float32.(rand(rng, 3, 100))
    res = kmeans(Lloyd(), X, 3; n_threads = 2, tol = 1e-6, verbose = false, rng = rng)

    @test typeof(res.totalcost) == Float32
    @test res.totalcost ≈ 14.133433f0
    @test res.converged
    @test res.iterations == 5
end

@testset "Lloyd test weighted X" begin
    rng = StableRNG(2020)
    X = rand(rng, 3, 100)
    weights = rand(rng, 100)

    init = sample(rng, 1:100, 10, replace = false)
    init = X[:, init]

    res = kmeans(Lloyd(), X, 10; weights = weights, init = init, n_threads = 1, tol = 1e-10, max_iters = 100, verbose = false, rng = rng)
    @test res.totalcost ≈ 2.3774024861034575
    @test res.converged
    @test res.iterations == 9

    rng = StableRNG(2020)
    X = rand(rng, 3, 100)
    weights = rand(rng, 100)

    res = kmeans(Lloyd(), X, 10; weights = weights, n_threads = 1, tol = 1e-10, max_iters = 100, verbose = false, rng = rng)
    @test res.totalcost ≈ 2.398132337904731
    @test res.converged
    @test res.iterations == 6

    rng = StableRNG(2020)
    X = rand(rng, 3, 100)
    weights = rand(rng, 100)

    res = kmeans(Lloyd(), X, 10; weights = weights, n_threads = 1, tol = 1e-10, max_iters = 100, verbose = false, rng = rng)
    @test res.totalcost ≈ 2.398132337904731
    @test res.converged
    @test res.iterations == 6
end

@testset "Lloyd metric support" begin
    Random.seed!(2020)
    X = [1. 2. 4.;]

    res = kmeans(Lloyd(), X, 2; tol = 1e-16, metric=Cityblock())

    @test res.assignments == [1, 1, 2]
    @test res.centers == [1.5 4.0]
    @test res.totalcost == 1.0
    @test res.converged

    Random.seed!(2020)
    X = rand(3, 100)

    res = kmeans(X, 2, tol = 1e-16, metric=Cityblock())
    @test res.totalcost ≈ 62.04045252895372
    @test res.converged
end

end # module
