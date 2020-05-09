module BenchDistance
using BenchmarkTools
using ParallelKMeans
using Distances
using Random

suite = BenchmarkGroup()

Random.seed!(2020)
X = rand(3, 100_000)
centroids = rand(3, 2)
d = fill(-Inf, 100_000)
suite["100kx3"] = @benchmarkable ParallelKMeans.chunk_colwise(d1, $X, $centroids, 1, nothing, Euclidean(), 1:100_000, 1) setup=(d1 = copy(d))

X = rand(10, 100_000)
centroids = rand(10, 2)
d = fill(-Inf, 100_000)
suite["100kx10"] = @benchmarkable ParallelKMeans.chunk_colwise(d1, $X, $centroids, 1, nothing, Euclidean(), 1:100_000, 1) setup=(d1 = copy(d))

end # module

BenchDistance.suite
