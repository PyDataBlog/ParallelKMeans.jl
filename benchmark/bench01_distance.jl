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
suite["100kx3"] = @benchmarkable ParallelKMeans.chunk_colwise($d, $X, $centroids, 1, nothing, 1:100_000, 1)

X = rand(10, 100_000)
centroids = rand(10, 2)
d = Vector{Float64}(undef, 100_000)
suite["100kx10"] = @benchmarkable ParallelKMeans.chunk_colwise($d, $X, $centroids, 1, nothing, 1:100_000, 1)

end # module

BenchDistance.suite
