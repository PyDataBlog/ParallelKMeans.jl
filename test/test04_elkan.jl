module TestElkan

using ParallelKMeans
using ParallelKMeans: MultiThread, LightElkan, update_centroids_dist!
using Test
using Random

@testset "centroid distances" begin
    centroids_dist = Matrix{Float64}(undef, 3, 3)
    centroids = [1.0 2.0 4.0; 2.0 1.0 3.0]
    update_centroids_dist!(centroids_dist, centroids)
    @test centroids_dist[1, 2] == centroids_dist[2, 1]
    @test centroids_dist[1, 3] == centroids_dist[3, 1]
    @test centroids_dist[2, 3] == centroids_dist[3, 2]
    @test centroids_dist[1, 2] == 0.5
    @test centroids_dist[1, 3] == 2.5
    @test centroids_dist[2, 3] == 2.0
    @test centroids_dist[1, 1] == 0.5
    @test centroids_dist[2, 2] == 0.5
    @test centroids_dist[3, 3] == 2.0
end

@testset "basic kmeans" begin
    X = [1. 2. 4.;]
    res = kmeans(LightElkan(), X, 1; tol = 1e-6, verbose = false)
    @test res.assignments == [1, 1, 1]
    @test res.centers[1] ≈ 2.3333333333333335
    @test res.totalcost ≈ 4.666666666666666
    @test res.converged

    res = kmeans(LightElkan(), X, 2; init = [1.0 4.0], tol = 1e-6, verbose = false)
    @test res.assignments == [1, 1, 2]
    @test res.centers ≈ [1.5 4.0]
    @test res.totalcost ≈ 0.5
    @test res.converged
end

@testset "no convergence yield last result" begin
    X = [1. 2. 4.;]
    res = kmeans(LightElkan(), X, 2; init = [1.0 4.0], tol = 1e-6, max_iters = 1, verbose = false)
    @test !res.converged
    @test res.totalcost ≈ 0.5
end

@testset "singlethread linear separation" begin
    Random.seed!(2020)

    X = rand(3, 100)
    res = kmeans(LightElkan(), X, 3; tol = 1e-6, verbose = false)

    @test res.totalcost ≈ 14.16198704459199
    @test res.converged
    @test res.iterations == 11
end


@testset "multithread linear separation quasi single thread" begin
    Random.seed!(2020)

    X = rand(3, 100)
    res = kmeans(LightElkan(), X, 3, MultiThread(1); tol = 1e-6, verbose = false)

    @test res.totalcost ≈ 14.16198704459199
    @test res.converged
end


@testset "multithread linear separation quasi two threads" begin
    Random.seed!(2020)

    X = rand(3, 100)
    res = kmeans(LightElkan(), X, 3, MultiThread(2); tol = 1e-6, verbose = false)

    @test res.totalcost ≈ 14.16198704459199
    @test res.converged
end

end # module
