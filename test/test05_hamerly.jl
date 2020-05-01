module TestHamerly

using ParallelKMeans
using ParallelKMeans: chunk_initialize, double_argmax
using Test
using StableRNGs
using Random
using Distances

@testset "initialize" begin
    X = permutedims([1.0 2; 2 1; 4 5; 6 6])
    centroids = permutedims([1.0 2; 4 5; 6 6])
    nrow, ncol = size(X)
    containers = ParallelKMeans.create_containers(Hamerly(), X, 3, nrow, ncol, 1)

    ParallelKMeans.chunk_initialize(Hamerly(), containers, centroids, X, nothing, Euclidean(), 1:ncol, 1)
    @test containers.lb == [18.0, 20.0, 5.0, 5.0]
    @test containers.ub == [0.0, 2.0, 0.0, 0.0]
end

@testset "double argmax" begin
    @test double_argmax([0.5, 0, 0]) == (1, 2, 0.5, 0.0)
end

@testset "singlethread linear separation" begin
    # with the same amount of iterations answer should be the same as in Lloyd case
    rng = StableRNG(2020)

    X = rand(rng, 3, 100)
    rng_orig = deepcopy(rng)
    res = kmeans(Hamerly(), X, 3; n_threads = 1, tol = 1e-10, max_iters = 4, verbose = false, rng = rng)

    @test res.totalcost ≈ 14.133433380466027
    @test !res.converged
    @test res.iterations == 4

    rng = deepcopy(rng_orig)
    res = kmeans(Hamerly(), X, 3; n_threads = 1, tol = 1e-10, max_iters = 1000, verbose = false, rng = rng)

    @test res.totalcost ≈ 14.133433380466027
    @test res.converged
    @test res.iterations == 5
end

@testset "multithread linear separation quasi two threads" begin
    rng = StableRNG(2020)

    X = rand(rng, 3, 100)
    rng_orig = deepcopy(rng)
    res = kmeans(Hamerly(), X, 3; n_threads = 2, tol = 1e-10, max_iters = 4, verbose = false, rng = rng)

    @test res.totalcost ≈ 14.133433380466027
    @test !res.converged
    @test res.iterations == 4

    rng = deepcopy(rng_orig)
    res = kmeans(Hamerly(), X, 3; n_threads = 2, tol = 1e-10, max_iters = 1000, verbose = false, rng = rng)

    @test res.totalcost ≈ 14.133433380466027
    @test res.converged
    @test res.iterations == 5
end

@testset "Hamerly Float32 support" begin
    rng = StableRNG(2020)

    X = Float32.(rand(rng, 3, 100))
    rng_orig = deepcopy(rng)
    res = kmeans(Hamerly(), X, 3; n_threads = 1, tol = 1e-6, verbose = false, rng = rng)

    @test typeof(res.totalcost) == Float32
    @test res.totalcost ≈ 14.133433f0
    @test res.converged
    @test res.iterations == 5

    rng = deepcopy(rng_orig)
    res = kmeans(Hamerly(), X, 3; n_threads = 2, tol = 1e-6, verbose = false, rng = rng)

    @test typeof(res.totalcost) == Float32
    @test res.totalcost ≈ 14.133433f0
    @test res.converged
    @test res.iterations == 5
end

@testset "Hamerly weights support" begin
    rng = StableRNG(2020)
    X = rand(rng, 3, 100)
    weights = rand(rng, 100)
    rng_orig = deepcopy(rng)

    baseline = kmeans(Lloyd(), X, 10; weights =  weights, tol = 1e-10, verbose = false, rng = rng)

    rng = deepcopy(rng_orig)
    res = kmeans(Hamerly(), X, 10; weights = weights, tol = 1e-10, verbose = false, rng = rng)
    @test res.totalcost ≈ baseline.totalcost
    @test res.converged
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Hamerly(), X, 10; weights = weights, n_threads = 2, tol = 1e-10, verbose = false, rng = rng)
    @test res.totalcost ≈ baseline.totalcost
    @test res.converged
    @test res.iterations == baseline.iterations
end


@testset "Hamerly metric support" begin
    Random.seed!(2020)
    X = [1. 2. 4.;]

    res = kmeans(Hamerly(), X, 2; tol = 1e-16, metric=Cityblock())

    @test res.assignments == [1, 1, 2]
    @test res.centers == [1.5 4.0]
    @test res.totalcost == 1.0
    @test res.converged

    Random.seed!(2020)
    X = rand(3, 100)

    res = kmeans(Hamerly(), X, 2, tol = 1e-16, metric=Cityblock())
    @test res.totalcost ≈ 62.04045252895372
    @test res.converged
end

end # module
