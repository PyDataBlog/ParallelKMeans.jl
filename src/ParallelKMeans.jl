module ParallelKMeans

using StatsBase
import Base.Threads: @spawn, @threads

export kmeans

abstract type CalculationMode end

# Single thread class to control the calculation type based on the CalculationMode
struct SingleThread <: CalculationMode
end

# Multi threaded implementation to control the calculation type based avaialble threads
struct MultiThread <: CalculationMode
    n::Int
end

# Get the number of avaialble threads for multithreading implementation
MultiThread() = MultiThread(Threads.nthreads())

# TODO here we mimic `Clustering` data structure, should thing how to integrate these
# two packages more closely.

"""
    ClusteringResult
Base type for the output of clustering algorithm.
"""
abstract type ClusteringResult end

# C is the type of centers, an (abstract) matrix of size (d x k)
# D is the type of pairwise distance computation from points to cluster centers
# WC is the type of cluster weights, either Int (in the case where points are
# unweighted) or eltype(weights) (in the case where points are weighted).
"""
    KmeansResult{C,D<:Real,WC<:Real} <: ClusteringResult
The output of [`kmeans`](@ref) and [`kmeans!`](@ref).
# Type parameters
 * `C<:AbstractMatrix{<:AbstractFloat}`: type of the `centers` matrix
 * `D<:Real`: type of the assignment cost
 * `WC<:Real`: type of the cluster weight
"""
struct KmeansResult{C<:AbstractMatrix{<:AbstractFloat},D<:Real,WC<:Real} <: ClusteringResult
    centers::C                 # cluster centers (d x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{D}           # cost of the assignments (n)
    counts::Vector{Int}        # number of points assigned to each cluster (k)
    wcounts::Vector{WC}        # cluster weights (k)
    totalcost::D               # total cost (i.e. objective)
    iterations::Int            # number of elapsed iterations
    converged::Bool            # whether the procedure converged
end

"""
    colwise!(target, x, y, mode)

Let X is a matrix `m x n` and Y is a vector of the length `m`. Then the `colwise!` function
computes distance between each column in X and Y and store result
in `target` array. Argument `mode` defines calculation mode, currently
following modes supported
- SingleThread()
- MultiThread()
"""
colwise!(target, x, y) = colwise!(target, x, y, SingleThread())

function colwise!(target, x, y, mode::SingleThread)
    @inbounds for j in axes(x, 2)
        res = 0.0
        for i in axes(x, 1)
            res += (x[i, j] - y[i])^2
        end
        target[j] = res
    end
end

"""
    spliiter(n, k)

Utility function, splits 1:n sequence to k chunks of approximately same size.
"""
function splitter(n, k)
    xz = Int.(ceil.(range(0, n, length = k+1)))
    return [xz[i]+1:xz[i+1] for i in 1:k]
end


function colwise!(target, x, y, mode::MultiThread)
    ncol = size(x, 2)

    ranges = splitter(ncol, mode.n)
    waiting_list = Task[]

    for i in 1:length(ranges) - 1
        push!(waiting_list, @spawn chunk_colwise!(target, x, y, ranges[i]))
    end

    chunk_colwise!(target, x, y, ranges[end])

    for i in 1:length(ranges) - 1
        wait(waiting_list[i])
    end

    target
end


"""
    chunk_colwise!(target, x, y, r)

Utility function for calculation of the colwise!(target, x, y, mode) function.
UnitRange argument `r` select subarray of original design matrix `x` that is going
to be processed.
"""
function chunk_colwise!(target, x, y, r)
    @inbounds for j in r
        res = 0.0
        for i in axes(x, 1)
            res += (x[i, j] - y[i])^2
        end
        target[j] = res
    end
end

"""
    smart_init(X, k; init="k-means++")

This function handles the random initialisation of the centroids from the
design matrix (X) and desired groups (k) that a user supplies.

`k-means++` algorithm is used by default with the normal random selection
of centroids from X used if any other string is attempted.

A tuple representing the centroids, number of rows, & columns respecitively
is returned.
"""
function smart_init(X::Array{Float64, 2}, k::Int, mode::T = SingleThread();
        init::String="k-means++") where {T <: CalculationMode}

    n_row, n_col = size(X)

    if init == "k-means++"

        # randonmly select the first centroid from the data (X)
        centroids = zeros(n_row, k)
        rand_indices = Vector{Int}(undef, k)
        rand_idx = rand(1:n_col)
        rand_indices[1] = rand_idx
        centroids[:, 1] .= X[:, rand_idx]
        distances = Vector{Float64}(undef, n_col)
        new_distances = Vector{Float64}(undef, n_col)

        # TODO: Add `colwise` function (or use it from `Distances` package)
        # compute distances from the first centroid chosen to all the other data points

        # flatten distances
        colwise!(distances, X, centroids[:, 1], mode)
        distances[rand_idx] = 0.0

        for i = 2:k
            # choose the next centroid, the probability for each data point to be chosen
            # is directly proportional to its squared distance from the nearest centroid
            r_idx = wsample(1:n_col, vec(distances))
            rand_indices[i] = r_idx
            centroids[:, i] .= X[:, r_idx]

            # no need for final distance update
            i == k && break

            # compute distances from the centroids to all data points
            colwise!(new_distances, X, centroids[:, i], mode)

            # and update the squared distance as the minimum distance to all centroid
            for i in 1:n_row
                distances[i, 1] = distances[i, 1] < new_distances[i, 1] ? distances[i, 1] : new_distances[i, 1]
            end
            distances[r_idx, 1] = 0.0
        end

    else
        # randomly select points from the design matrix as the initial centroids
        # TODO change rand to sample
        rand_indices = rand(1:n_col, k)
        centroids = X[:, rand_indices]
    end

    return (centroids = centroids, indices = rand_indices)
end


"""
    sum_of_squares(x, labels, centre, k)

This function computes the total sum of squares based on the assigned (labels)
design matrix(x), centroids (centre), and the number of desired groups (k).

A Float type representing the computed metric is returned.
"""
function sum_of_squares(x::Array{Float64,2}, labels::Array{Int64,1}, centre::Array)
    s = 0.0

    @inbounds for j in axes(x, 2)
        for i in axes(x, 1)
            s += (x[i, j] - centre[i, labels[j]])^2
        end
    end

    return s
end


"""
    Kmeans(design_matrix, k; k_init="k-means++", max_iters=300, tol=1e-4, verbose=true)

This main function employs the K-means algorithm to cluster all examples
in the training data (design_matrix) into k groups using either the
`k-means++` or random initialisation technique for selecting the initial
centroids.

At the end of the number of iterations specified (max_iters), convergence is
achieved if difference between the current and last cost objective is
less than the tolerance level (tol). An error is thrown if convergence fails.

Details of operations can be either printed or not by setting verbose accordingly.

A tuple representing labels, centroids, and sum_squares respectively is returned.
"""
function kmeans(design_matrix::Array{Float64, 2}, k::Int, mode::T = SingleThread();
                k_init::String = "k-means++", max_iters::Int = 300, tol = 1e-4, verbose::Bool = true, init = nothing) where {T <: CalculationMode}
    nrow, ncol = size(design_matrix)
    centroids = init == nothing ? smart_init(design_matrix, k, mode, init=k_init).centroids : init
    new_centroids = similar(centroids)

    labels = Vector{Int}(undef, ncol)
    centroids_cnt = Vector{Int}(undef, k)

    J_previous = Inf64
    totalcost = Inf

    # nearest_neighbour = Array{Float64, 2}(undef, size(design_matrix, 1), size(centroids, 1))
    # Update centroids & labels with closest members until convergence
    for iter = 1:max_iters
        J = update_centroids!(centroids, new_centroids, centroids_cnt, labels, design_matrix, mode)
        J /= ncol

        if verbose
            # Show progress and terminate if J stopped decreasing.
            println("Iteration $iter: Jclust = $J")
        end

        # Final Step: Check for convergence
        if (iter > 1) & (abs(J - J_previous) < (tol * J))

            totalcost = sum_of_squares(design_matrix, labels, centroids)

            # Terminate algorithm with the assumption that K-means has converged
            if verbose
                println("Successfully terminated with convergence.")
            end

            # TODO empty vectors should be calculated
            # TODO Float64 type definitions is too restrictive, should be relaxed
            # especially during GPU related development
            return KmeansResult(centroids, labels, Float64[], Int[], Float64[], totalcost, iter, true)

        elseif (iter == max_iters) & (abs(J - J_previous) > (tol * J))
            return KmeansResult(centroids, labels, Float64[], Int[], Float64[], totalcost, iter + 1, false)
        end

        J_previous = J
    end
end

function update_centroids!(centroids, new_centroids, centroids_cnt, labels,
        design_matrix, mode::SingleThread)

    r = axes(design_matrix, 2)
    J = chunk_update_centroids!(centroids, new_centroids, centroids_cnt, labels,
        design_matrix, r, mode)

    centroids .= new_centroids ./ centroids_cnt'

    return J
end

function update_centroids!(centroids, new_centroids, centroids_cnt, labels,
        design_matrix, mode::MultiThread)
    mode.n == 1 && return update_centroids!(centroids, new_centroids, centroids_cnt, labels,
            design_matrix, SingleThread())

    ncol = size(design_matrix, 2)

    ranges = splitter(ncol, mode.n)

    waiting_list = Vector{Task}(undef, mode.n - 1)

    for i in 1:length(ranges) - 1
        waiting_list[i] = @spawn chunk_update_centroids!(centroids, new_centroids, centroids_cnt, labels,
            design_matrix, ranges[i], mode)
    end

    J = chunk_update_centroids!(centroids, new_centroids, centroids_cnt, labels,
        design_matrix, ranges[end], mode)

    J += sum(fetch.(waiting_list))

    centroids .= new_centroids ./ centroids_cnt'

    return J
end


function chunk_update_centroids!(centroids, new_centroids, centroids_cnt, labels,
    design_matrix, r, mode::T = SingleThread()) where {T <: CalculationMode}

    new_centroids .= 0.0
    centroids_cnt .= 0
    J = 0.0
    @inbounds for i in r
        min_distance = Inf
        label = 1
        for k in axes(centroids, 2)
            distance = 0.0
            for j in axes(design_matrix, 1)
                distance += (design_matrix[j, i] - centroids[j, k])^2
            end
            label = min_distance > distance ? k : label
            min_distance = min_distance > distance ? distance : min_distance
        end
        labels[i] = label
        centroids_cnt[label] += 1
        for j in axes(design_matrix, 1)
            new_centroids[j, label] += design_matrix[j, i]
        end
        J += min_distance
    end
    # centroids .= new_centroids ./ centroids_cnt'

    return J
end

end # module
