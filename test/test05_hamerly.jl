module TestHamerly

using ParallelKMeans
using ParallelKMeans: chunk_initialize, double_argmax
using Test
using Random

@testset "initialize" begin
    X = permutedims([1.0 2; 2 1; 4 5; 6 6])
    centroids = permutedims([1.0 2; 4 5; 6 6])
    nrow, ncol = size(X)
    containers = ParallelKMeans.create_containers(Hamerly(), 3, nrow, ncol, 1)

    ParallelKMeans.chunk_initialize(Hamerly(), containers, centroids, X, 1:ncol, 1)
    @test containers.lb == [18.0, 20.0, 5.0, 5.0]
    @test containers.ub == [0.0, 2.0, 0.0, 0.0]
end

@testset "double argmax" begin
    @test double_argmax([0.5, 0, 0]) == (1, 2, 0.5, 0.0)
end

@testset "singlethread linear separation" begin
    # with the same amount of iterations answer should be the same as in Lloyd case
    Random.seed!(2020)

    X = rand(3, 100)
    res = kmeans(Hamerly(), X, 3; n_threads = 1, tol = 1e-10, max_iters = 10, verbose = false)

    @test res.totalcost ≈ 14.16198704459199
    @test !res.converged
    @test res.iterations == 10

    Random.seed!(2020)
    X = rand(3, 100)
    res = kmeans(Hamerly(), X, 3; n_threads = 1, tol = 1e-10, max_iters = 1000, verbose = false)

    @test res.totalcost ≈ 14.161987044591992
    @test res.converged
    @test res.iterations == 11
end

@testset "multithread linear separation quasi two threads" begin
    Random.seed!(2020)

    X = rand(3, 100)
    res = kmeans(Hamerly(), X, 3; n_threads = 2, tol = 1e-10, max_iters = 10, verbose = false)

    @test res.totalcost ≈ 14.16198704459199
    @test !res.converged
    @test res.iterations == 10

    Random.seed!(2020)
    X = rand(3, 100)
    res = kmeans(Hamerly(), X, 3; n_threads = 2, tol = 1e-10, max_iters = 1000, verbose = false)

    @test res.totalcost ≈ 14.161987044591992
    @test res.converged
    @test res.iterations == 11
end

end # module
