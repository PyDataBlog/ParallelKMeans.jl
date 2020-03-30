module TestVerbosity

using ParallelKMeans
using Random
using Test
using Suppressor


@testset "Testing verbosity of implementation" begin
    Random.seed!(2020)
    X = rand(4, 150)
    Random.seed!(2020)
    # Capture output and compare
    r = @capture_out kmeans(X, 3; n_threads=1, max_iters=1, verbose=true)
    @test r == "Iteration 1: Jclust = 0.31023197229652544\n"
    
end

end # module

