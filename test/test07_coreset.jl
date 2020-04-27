module TestCoreset

using ParallelKMeans
using Test
using Random

@testset "basic coresets" begin
    Random.seed!(2020)
    X = rand(3, 100)

    res = kmeans(Coreset(20), X, 10, tol = 1e-10, verbose = false, n_threads = 1)
    @test res.converged
    @test res.iterations == 4
    @test res.totalcost ≈ 7.667588608178126

    Random.seed!(2020)
    X = rand(3, 100)

    res = kmeans(Coreset(20), X, 10, tol = 1e-10, verbose = false, n_threads = 2)
    @test res.converged
    @test res.iterations == 4
    @test res.totalcost ≈ 7.667588608178126

    Random.seed!(2020)
    X = rand(3, 100)

    res = kmeans(Coreset(20, Lloyd()), X, 10, tol = 1e-10, verbose = false, n_threads = 1)
    @test res.converged
    @test res.iterations == 4
    @test res.totalcost ≈ 7.667588608178126
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
