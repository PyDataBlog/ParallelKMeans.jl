module TestKMeans

using ParallelKMeans
using ParallelKMeans: MultiThread
using Test
using Random

@testset "singlethread linear separation" begin
    Random.seed!(2020)

    X = rand(100, 3)
    labels, centroids, sum_squares = kmeans(X, 3; tol = 1e-10, verbose = false)

    @test sum_squares ≈ 14.964882850452971
end


@testset "multithread linear separation" begin
    Random.seed!(2020)

    X = rand(100, 3)
    labels, centroids, sum_squares = kmeans(X, 3, MultiThread(); tol = 1e-10, verbose = false)

    @test sum_squares ≈ 14.964882850452971
end

end # module
