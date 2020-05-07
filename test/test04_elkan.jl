module TestElkan

using ParallelKMeans
using Test
using StableRNGs

@testset "basic kmeans elkan" begin
    X = [1. 2. 4.;]
    res = kmeans(Elkan(), X, 1; n_threads = 1, tol = 1e-6, verbose = false)
    @test res.assignments == [1, 1, 1]
    @test res.centers[1] ≈ 2.3333333333333335
    @test res.totalcost ≈ 4.666666666666666
    @test res.converged

    res = kmeans(Elkan(), X, 2; n_threads = 1, init = [1.0 4.0], tol = 1e-6, verbose = false)
    @test res.assignments == [1, 1, 2]
    @test res.centers ≈ [1.5 4.0]
    @test res.totalcost ≈ 0.5
    @test res.converged
end

@testset "elkan no convergence yield last result" begin
    X = [1. 2. 4.;]
    res = kmeans(Elkan(), X, 2; n_threads = 1, init = [1.0 4.0], tol = 1e-6, max_iters = 1, verbose = false)
    @test !res.converged
    @test res.totalcost ≈ 0.5
end

@testset "elkan singlethread linear separation" begin
    rng = StableRNG(2020)

    X = rand(rng, 3, 100)
    res = kmeans(Elkan(), X, 3; n_threads = 1, tol = 1e-10, max_iters = 4, verbose = false, rng = rng)

    @test res.totalcost ≈ 14.133433380466027
    @test !res.converged
    @test res.iterations == 4
end

@testset "elkan multithread linear separation quasi two threads" begin
    rng = StableRNG(2020)

    X = rand(rng, 3, 100)
    res = kmeans(Elkan(), X, 3; n_threads = 2, tol = 1e-6, verbose = false, rng = rng)

    @test res.totalcost ≈ 14.133433380466027
    @test res.converged
end

@testset "Elkan Float32 support" begin
    rng = StableRNG(2020)

    X = Float32.(rand(rng, 3, 100))

    res = kmeans(Elkan(), X, 3; n_threads = 1, tol = 1e-6, verbose = false, rng = rng)

    @test typeof(res.totalcost) == Float32
    @test res.totalcost ≈ 14.133433f0
    @test res.converged
    @test res.iterations == 5

    rng = StableRNG(2020)
    X = Float32.(rand(rng, 3, 100))
    res = kmeans(Elkan(), X, 3; n_threads = 2, tol = 1e-6, verbose = false, rng = rng)

    @test typeof(res.totalcost) == Float32
    @test res.totalcost ≈ 14.133433f0
    @test res.converged
    @test res.iterations == 5
end

@testset "Elkan weights support" begin
    rng = StableRNG(2020)
    X = rand(rng, 3, 100)
    weights = rand(rng, 100)
    rng_orig = deepcopy(rng)
    baseline = kmeans(Lloyd(), X, 10; weights = weights, tol = 1e-10, verbose = false, rng = rng)

    rng = deepcopy(rng_orig)
    res = kmeans(Elkan(), X, 10; weights = weights, tol = 1e-10, verbose = false, rng = rng)
    @test res.totalcost ≈ baseline.totalcost
    @test res.converged
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Elkan(), X, 10; weights = weights, n_threads = 2, tol = 1e-10, verbose = false, rng = rng)
    @test res.totalcost ≈ baseline.totalcost
    @test res.converged
    @test res.iterations == baseline.iterations
end

end # module
