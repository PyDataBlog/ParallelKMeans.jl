module BenchDistance
using BenchmarkTools
using ParallelKMeans
using Distances
using Random

suite = BenchmarkGroup()

Random.seed!(2020)
X = rand(3, 100_000)
centroids = rand(3, 2)
d = Vector{Float64}(undef, 100_000)
suite["100kx3"] = @benchmarkable ParallelKMeans.colwise!($d, $X, $centroids)

X = rand(10, 100_000)
centroids = rand(10, 2)
d = Vector{Float64}(undef, 100_000)
suite["100kx10"] = @benchmarkable ParallelKMeans.colwise!($d, $X, $centroids)

# for reference
metric = SqEuclidean()
#suite["100kx10_distances"] = @benchmarkable Distances.colwise!($d, $metric, $X, $centroids)
dist = Distances.pairwise(metric, X, centroids, dims = 2)
min = minimum(dist, dims=2)
suite["100kx10_distances"] = @benchmarkable $d = min
end # module

BenchDistance.suite
