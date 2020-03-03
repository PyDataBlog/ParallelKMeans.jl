using Clustering
using ParallelKMeans
using Plots
using BenchmarkTools
using TimerOutputs
using Random
using ProgressMeter

# Create a TimerOutput, this is the main type that keeps track of everything.
const to = TimerOutput()

Random.seed!(2020)
X = rand(60, 1_000_000);
# Timed assingments
global a = Float64[]
global b = Float64[]
global c = Float64[]

p = Progress(9, 10, "Computing clustering...")
@timeit to "Clustering" begin
    for i in 2:10
        @timeit to "$i clusters" push!(a, Clustering.kmeans(X, i, tol=1e-6, maxiter=300).totalcost)
        next!(p)
    end
end

p = Progress(9, 10, "Computing singlethreaded ParallelKMeans...")
@timeit to "PKMeans Singlethread" begin
    for i in 2:10
        @timeit to "$i clusters" push!(b, ParallelKMeans.kmeans(X, i, tol=1e-6, max_iters=300, verbose=false).totalcost)
        next!(p)
    end
end

p = Progress(9, 10, "Computing multithreaded ParallelKMeans...")
@timeit to "PKMeans Multithread" begin
    for i in 2:10
        @timeit to "$i clusters" push!(c, ParallelKMeans.kmeans(X, i, ParallelKMeans.MultiThread(), tol=1e-6, max_iters=300, verbose=false).totalcost)
        next!(p)
    end
end

plot(a, label="Clustering.jl")
plot!(b, label="Single-Thread ParallelKmeans")
plot!(c, label="Multi-Thread ParallelKmeans")

print(to)
