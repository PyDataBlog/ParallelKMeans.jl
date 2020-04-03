module BenchKMeans
using Random
using ParallelKMeans
using BenchmarkTools

suite = BenchmarkGroup()

Random.seed!(2020)
X = rand(10, 100_000)

centroids3 = ParallelKMeans.smart_init(X, 3, 1, init="kmeans++").centroids
centroids10 = ParallelKMeans.smart_init(X, 10, 1, init="kmeans++").centroids

suite["10x100_000x3x1     Lloyd"] = @benchmarkable kmeans($X, 3, init = $centroids3, n_threads = 1, verbose = false, tol = 1e-6, max_iters = 1000)
suite["10x100_000x3x1  Hammerly"] = @benchmarkable kmeans(Hamerly(), $X, 3, init = $centroids3, n_threads = 1, verbose = false, tol = 1e-6, max_iters = 1000)

suite["10x100_000x3x2     Lloyd"] = @benchmarkable kmeans($X, 3, init = $centroids3, n_threads = 2, verbose = false, tol = 1e-6, max_iters = 1000)
suite["10x100_000x3x2  Hammerly"] = @benchmarkable kmeans(Hamerly(), $X, 3, init = $centroids3, n_threads = 2, verbose = false, tol = 1e-6, max_iters = 1000)

suite["10x100_000x10x1    Lloyd"] = @benchmarkable kmeans($X, 10, init = $centroids10, n_threads = 1, verbose = false, tol = 1e-6, max_iters = 1000)
suite["10x100_000x10x1 Hammerly"] = @benchmarkable kmeans(Hamerly(), $X, 10, init = $centroids10, n_threads = 1, verbose = false, tol = 1e-6, max_iters = 1000)

suite["10x100_000x10x2    Lloyd"] = @benchmarkable kmeans($X, 10, init = $centroids10, n_threads = 2, verbose = false, tol = 1e-6, max_iters = 1000)
suite["10x100_000x10x2 Hammerly"] = @benchmarkable kmeans(Hamerly(), $X, 10, init = $centroids10, n_threads = 2, verbose = false, tol = 1e-6, max_iters = 1000)

end # module

BenchKMeans.suite
