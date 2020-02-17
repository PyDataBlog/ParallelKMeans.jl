module TestKMeans
using ParallelKMeans
using Test
using Random

@testset "linear separation" begin
    Random.seed!(2020)

    X = rand(100, 3)
    labels, centroids, sum_squares = kmeans(X, 3; tol = 1e-10, verbose = false)

    # for future reference: Clustering shows here 14.964882850452984
    # guess they use better initialisation. For now we will use own
    # value
    @test sum_squares ≈ 15.314823028363763
end

end # module
