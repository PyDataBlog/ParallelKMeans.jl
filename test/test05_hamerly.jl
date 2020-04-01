module TestHamerly

using ParallelKMeans
using ParallelKMeans: initialize!, double_argmax
using Test
using Random

@testset "initialize" begin
    X = permutedims([1.0 2; 2 1; 4 5; 6 6])
    centroids = permutedims([1.0 2; 4 5; 6 6])
    nrow, ncol = size(X)
    containers = ParallelKMeans.create_containers(Hamerly(), 3, nrow, ncol, 1)

    ParallelKMeans.initialize!(Hamerly(), containers, centroids, X, 1)
    @test containers.lb == [18.0, 20.0, 5.0, 5.0]
    @test containers.ub == [0.0, 2.0, 0.0, 0.0]
end

@testset "double argmax" begin
    @test double_argmax([0.5, 0, 0]) == (1, 2)
end

end # module
