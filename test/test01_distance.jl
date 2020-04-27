module TestDistance
using ParallelKMeans: chunk_colwise, @parallelize
using Distances: Euclidean
using Test

@testset "naive singlethread colwise" begin
    X = [1.0 3.0 4.0; 2.0 5.0 6.0]
    y = permutedims([1.0, 2.0]')
    ncol = size(X, 2)
    r = fill(Inf, ncol)
    n_threads = 1

    @parallelize n_threads ncol chunk_colwise(Euclidean(), r, X, y, 1, nothing)
    @test all(r .≈ [0.0, 13.0, 25.0])
end

@testset "multithread colwise" begin
    X = [1.0 3.0 4.0; 2.0 5.0 6.0]
    y = permutedims([1.0, 2.0]')
    ncol = size(X, 2)
    r = fill(Inf, ncol)
    n_threads = 2

    @parallelize n_threads ncol chunk_colwise(Euclidean(), r, X, y, 1, nothing)

    @test all(r .≈ [0.0, 13.0, 25.0])
end

end # module
