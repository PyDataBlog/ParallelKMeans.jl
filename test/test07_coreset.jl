module TestCoreset

using ParallelKMeans
using Test
using StableRNGs
using Distances


@testset "basic coresets" begin
    rng = StableRNG(2020)
    X = rand(rng, 3, 100)
    rng_orig = deepcopy(rng)

    baseline = kmeans(Coreset(20), X, 10, tol = 1e-10, verbose = false, n_threads = 1, rng = rng)
    @test baseline.converged
    @test baseline.iterations == 4
    @test baseline.totalcost ≈ 7.870212645990514

    rng = deepcopy(rng_orig)
    res = kmeans(Coreset(20), X, 10, tol = 1e-10, verbose = false, n_threads = 2, rng = rng)
    @test res.converged
    @test res.iterations == baseline.iterations
    @test res.totalcost ≈ baseline.totalcost

    rng = deepcopy(rng_orig)
    res = kmeans(Coreset(20, Lloyd()), X, 10, tol = 1e-10, verbose = false, n_threads = 1, rng = rng)
    @test res.converged
    @test res.iterations == baseline.iterations
    @test res.totalcost ≈ baseline.totalcost
end

@testset "Coreset possible interfaces" begin
    alg = Coreset()
    @test alg.m == 100
    @test alg.alg == Lloyd()

    alg = Coreset(m = 200)
    @test alg.m == 200

    alg = Coreset(alg = Hamerly())
    @test alg.alg == Hamerly()

    alg = Coreset(200)
    @test alg.m == 200

    alg = Coreset(Hamerly())
    @test alg.alg == Hamerly()
end

@testset "Coreset metric support" begin
    rng = StableRNG(2020)
    X = [1. 2. 4.;]

    res = kmeans(Coreset(), X, 2; tol = 1e-16, metric=Cityblock(), rng = rng)

    @test res.assignments == [2, 2, 1]
    @test res.centers ≈ [4.0 1.4865168535972686]
    @test res.totalcost == 1.0
    @test res.converged


    rng = StableRNG(2020)
    X = rand(rng, 3, 100)
    rng_orig = deepcopy(rng)

    baseline = kmeans(Lloyd(), X, 10, tol = 1e-10, metric=Cityblock(), rng = rng, n_threads = 1)
    rng = deepcopy(rng_orig)
    res = kmeans(Coreset(), X, 10; tol = 1e-10, metric = Cityblock(), rng = rng, n_threads = 1)

    @test res.converged == baseline.converged
    @test res.iterations == baseline.iterations
    @test floor(res.totalcost - baseline.totalcost) ≤ 1

end

end # module
