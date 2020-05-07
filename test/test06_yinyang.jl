module TestYinyang

using ParallelKMeans
using Test
using StableRNGs

@testset "basic kmeans Yinyang" begin
    X = [1. 2. 4.;]
    res = kmeans(Yinyang(false), X, 1; n_threads = 1, tol = 1e-6, verbose = false)
    @test res.assignments == [1, 1, 1]
    @test res.centers[1] ≈ 2.3333333333333335
    @test res.totalcost ≈ 4.666666666666666
    @test res.converged

    res = kmeans(Yinyang(false), X, 2; n_threads = 1, init = [1.0 4.0], tol = 1e-6, verbose = false)
    @test res.assignments == [1, 1, 2]
    @test res.centers ≈ [1.5 4.0]
    @test res.totalcost ≈ 0.5
    @test res.converged
end

@testset "Yinyang no convergence yield last result" begin
    X = [1. 2. 4.;]
    res = kmeans(Yinyang(false), X, 2; n_threads = 1, init = [1.0 4.0], tol = 1e-6, max_iters = 1, verbose = false)
    @test !res.converged
    @test res.totalcost ≈ 0.5
end

@testset "Yinyang singlethread linear separation" begin
    rng = StableRNG(2020)

    X = rand(rng, 3, 100)
    res = kmeans(Yinyang(false), X, 3; n_threads = 1, tol = 1e-10, max_iters = 4, verbose = false, rng = rng)

    @test res.totalcost ≈ 14.133433380466027
    @test !res.converged
    @test res.iterations == 4
end

@testset "Yinyang multithread linear separation quasi two threads" begin
    rng = StableRNG(2020)

    X = rand(rng, 3, 100)
    res = kmeans(Yinyang(false), X, 3; n_threads = 2, tol = 1e-6, verbose = false, rng = rng)

    @test res.totalcost ≈ 14.133433380466027
    @test res.converged
end

@testset "Yinyang different modes" begin
    rng = StableRNG(2020)

    X = rand(rng, 3, 100)
    init = ParallelKMeans.smart_init(X, 20, 1, nothing, rng).centroids
    rng_orig = deepcopy(rng)
    baseline = kmeans(Lloyd(), X, 20, init = init, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000, rng = rng)

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(false), X, 20, init = init, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000, rng = rng)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(), X, 20, init = init, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000, rng = rng)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(10), X, 20, init = init, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000, rng = rng)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(7), X, 20, init = init, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000, rng = rng)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(false), X, 20, init = init, tol = 1e-10, n_threads = 2, verbose = false, max_iters = 1000, rng = rng)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(), X, 20, init = init, tol = 1e-10, n_threads = 2, verbose = false, max_iters = 1000, rng = rng)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(10), X, 20, init = init, tol = 1e-10, n_threads = 2, verbose = false, max_iters = 1000, rng = rng)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(7), X, 20, init = init, tol = 1e-10, n_threads = 2, verbose = false, max_iters = 1000, rng = rng)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations
end

@testset "Yinyang Float32 support" begin
    rng = StableRNG(2020)

    X = Float32.(rand(rng, 3, 100))
    rng_orig = deepcopy(rng)
    baseline = kmeans(Lloyd(), X, 20, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000, rng = rng)

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(5), X, 20, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000, rng = rng)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations
    @test typeof(res.totalcost) == Float32

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(5), X, 20, tol = 1e-10, n_threads = 2, verbose = false, max_iters = 1000, rng = rng)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations
    @test typeof(res.totalcost) == Float32
end

@testset "Yinyang weights support" begin
    rng = StableRNG(2020)
    X = rand(rng, 3, 100)
    weights = rand(rng, 100)
    rng_orig = deepcopy(rng)
    baseline = kmeans(Lloyd(), X, 10; weights = weights, tol = 1e-10, verbose = false, rng = rng)

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(), X, 10, weights = weights, tol = 1e-10, verbose = false, rng = rng)
    @test res.totalcost ≈ baseline.totalcost
    @test res.converged
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(), X, 10, weights = weights, n_threads = 2, tol = 1e-10, verbose = false, rng = rng)
    @test res.totalcost ≈ baseline.totalcost
    @test res.converged
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(5), X, 10, weights = weights, tol = 1e-10, verbose = false, rng = rng)
    @test res.totalcost ≈ baseline.totalcost
    @test res.converged
    @test res.iterations == baseline.iterations

    rng = deepcopy(rng_orig)
    res = kmeans(Yinyang(5), X, 10, weights = weights, n_threads = 2, tol = 1e-10, verbose = false, rng = rng)
    @test res.totalcost ≈ baseline.totalcost
    @test res.converged
    @test res.iterations == baseline.iterations
end

@testset "Yinyang possible interfaces" begin
    alg = Yinyang()
    @test alg.auto
    @test alg.group_size == 7

    alg = Yinyang(group_size = 10)
    @test alg.group_size == 10

    alg = Yinyang(auto = false)
    @test !alg.auto

    alg = Yinyang(10)
    @test alg.group_size == 10

    alg = Yinyang(false)
    @test !alg.auto
end

end # module
