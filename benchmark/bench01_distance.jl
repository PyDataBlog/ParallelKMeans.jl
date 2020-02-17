module BenchDistance
using BenchmarkTools
using ParallelKMeans
using Distances
using Random

suite = BenchmarkGroup()

Random.seed!(2020)
X = rand(100_000, 3)
centroids = rand(2, 3)
d = rand(100_000, 2)
suite["100kx3"] = @benchmarkable ParallelKMeans.pairwise!($d, $X, $centroids)

X = rand(100_000, 10)
centroids = rand(2, 10)
d = rand(100_000, 2)
suite["100kx10"] = @benchmarkable ParallelKMeans.pairwise!($d, $X, $centroids)

# for reference
metric = SqEuclidean()
suite["100kx10_distances"] = @benchmarkable Distances.pairwise!($d, $metric, $X, $centroids, dims = 1)

end # module

BenchDistance.suite
