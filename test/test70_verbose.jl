module TestVerbosity

using ParallelKMeans
using StableRNGs
using Test
using Suppressor


@testset "LLoyd: Testing verbosity of implementation" begin
    rng = StableRNG(2020)
    X = rand(rng, 4, 150)

    # Capture output and compare
    r = @capture_out kmeans(Lloyd(), X, 3; n_threads=1, max_iters=1, verbose=true, rng = rng)
    @test startswith(r, "Iteration 1: Jclust = 41.94858243")
end

@testset "Hamerly: Testing verbosity of implementation" begin
    rng = StableRNG(2020)
    X = rand(rng, 4, 150)

    # Capture output and compare
    r = @capture_out kmeans(Hamerly(), X, 3; n_threads=1, max_iters=1, verbose=true, rng = rng)
    @test startswith(r, "Iteration 1: Jclust = 41.94858243")
end

@testset "Elkan: Testing verbosity of implementation" begin
    rng = StableRNG(2020)
    X = rand(rng, 4, 150)

    # Capture output and compare
    r = @capture_out kmeans(Elkan(), X, 3; n_threads=1, max_iters=1, verbose=true, rng = rng)
    @test startswith(r, "Iteration 1: Jclust = 41.94858243")
end

@testset "Yinyang: Testing verbosity of implementation" begin
    rng = StableRNG(2020)
    X = rand(rng, 4, 150)

    # Capture output and compare
    r = @capture_out kmeans(Yinyang(), X, 3; n_threads=1, max_iters=1, verbose=true, rng = rng)
    @test startswith(r, "Iteration 1: Jclust = 74.7253379541")
end

@testset "Coreset: Testing verbosity of implementation" begin
    rng = StableRNG(2020)
    X = rand(rng, 4, 150)

    # Capture output and compare
    r = @capture_out kmeans(Coreset(), X, 3; n_threads=1, max_iters=1, verbose=true, rng = rng)
    # This test is broken on 1.5 dev, see https://github.com/rfourquet/StableRNGs.jl/issues/3
    # @test startswith(r, "Iteration 1: Jclust = 32.8028409136")
end

end # module
