module TestCoreset

using ParallelKMeans
using Test
using StableRNGs

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

end # module
