module TestYinyang

using ParallelKMeans
using Test
using Random

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
    Random.seed!(2020)

    X = rand(3, 100)
    res = kmeans(Yinyang(false), X, 3; n_threads = 1, tol = 1e-10, max_iters = 10, verbose = false)

    @test res.totalcost ≈ 14.16198704459199
    @test !res.converged
    @test res.iterations == 10
end

@testset "Yinyang multithread linear separation quasi two threads" begin
    Random.seed!(2020)

    X = rand(3, 100)
    res = kmeans(Yinyang(false), X, 3; n_threads = 2, tol = 1e-6, verbose = false)

    @test res.totalcost ≈ 14.16198704459199
    @test res.converged
end

@testset "Yinyang different modes" begin
    Random.seed!(2020)
    X = rand(3, 100)
    init = ParallelKMeans.smart_init(X, 20).centroids
    baseline = kmeans(Lloyd(), X, 20, init = init, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000)

    res = kmeans(Yinyang(false), X, 20, init = init, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    res = kmeans(Yinyang(), X, 20, init = init, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    res = kmeans(Yinyang(10), X, 20, init = init, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    res = kmeans(Yinyang(7), X, 20, init = init, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    res = kmeans(Yinyang(false), X, 20, init = init, tol = 1e-10, n_threads = 2, verbose = false, max_iters = 1000)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    res = kmeans(Yinyang(), X, 20, init = init, tol = 1e-10, n_threads = 2, verbose = false, max_iters = 1000)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    res = kmeans(Yinyang(10), X, 20, init = init, tol = 1e-10, n_threads = 2, verbose = false, max_iters = 1000)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations

    res = kmeans(Yinyang(7), X, 20, init = init, tol = 1e-10, n_threads = 2, verbose = false, max_iters = 1000)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations
end

@testset "Yinyang Float32 support" begin
    Random.seed!(2020)
    X = Float32.(rand(3, 100))
    baseline = kmeans(Lloyd(), X, 20, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000)

    Random.seed!(2020)
    X = Float32.(rand(3, 100))

    res = kmeans(Yinyang(5), X, 20, tol = 1e-10, n_threads = 1, verbose = false, max_iters = 1000)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations
    @test typeof(res.totalcost) == Float32

    Random.seed!(2020)
    X = Float32.(rand(3, 100))

    res = kmeans(Yinyang(5), X, 20, tol = 1e-10, n_threads = 2, verbose = false, max_iters = 1000)
    @test res.converged
    @test res.totalcost ≈ baseline.totalcost
    @test res.assignments == baseline.assignments
    @test res.centers ≈ baseline.centers
    @test res.iterations == baseline.iterations
    @test typeof(res.totalcost) == Float32
end

end # module
