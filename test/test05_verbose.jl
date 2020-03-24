module TestVerbosity

using ParallelKMeans
using Random
using Test


@testset "Testing verbosity of implementation" begin
    Random.seed!(2020)
    X = rand(4, 150)
    Random.seed!(2020)
    @test_logs (:info, "Iteration 1: Jclust = 0.31023197229652544") kmeans(X, 3, n_threads=1, max_iters=1)
    
end

end # module

