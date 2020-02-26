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

"""
    pairwise!(target, x, y, mode)

Let X and Y respectively have m and n columns. Then the `pairwise!` function
computes distances between each pair of columns in X and Y and store result
in `target` array. Argument `mode` defines calculation mode, currently
following modes supported
- SingleThread()
- MultiThread()
"""
pairwise!(target, x, y) = pairwise!(target, x, y, SingleThread())

function pairwise!(target, x, y, mode::SingleThread)
    ncol = size(x, 2)

    @inbounds for k in axes(y, 1)
        for i in axes(x, 1)
            target[i, k] = (x[i, 1] - y[k, 1])^2
        end

        for j in 2:ncol
            for i in axes(x, 1)
                target[i, k] += (x[i, j] - y[k, j])^2
            end
        end
    end
    target
end

"""
    divider(n, k)

Utility function, splits 1:n sequence to k chunks of approximately same size.
"""
function divider(n, k)
    d = div(n, k)
    xz = vcat(collect((0:k-1) * d), n)
    return [t[1]:t[2] for t in zip(xz[1:end-1] .+ 1, xz[2:end])]
end


function pairwise!(target, x, y, mode::MultiThread)
    ncol = size(x, 2)
    nrow = size(x, 1)

    ranges = divider(nrow, mode.n)
    waiting_list = Task[]

    for i in 1:length(ranges) - 1
        push!(waiting_list, @spawn inner_pairwise!(target, x, y, ranges[i]))
    end

    inner_pairwise!(target, x, y, ranges[end])

    for i in 1:length(ranges) - 1
        wait(waiting_list[i])
    end

    target
end


"""
    inner_pairwise!(target, x, y, r)

Utility function for calculation of [pairwise!(target, x, y, mode)](@ref) function.
UnitRange argument `r` select subarray of original design matrix `x` that is going
to be processed.
"""
function inner_pairwise!(target, x, y, r)
    ncol = size(x, 2)

    @inbounds for k in axes(y, 1)
        for i in r
            target[i, k] = (x[i, 1] - y[k, 1])^2
        end

        for j in 2:ncol
            for i in r
                target[i, k] += (x[i, j] - y[k, j])^2
            end
        end
    end
    target
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
        centroids = zeros(k, n_col)
        rand_indices = Vector{Int}(undef, k)
        rand_idx = rand(1:n_row)
        rand_indices[1] = rand_idx
        centroids[1, :] .= X[rand_idx, :]
        centroids[k, :] .= 0.0
        distances = Array{Float64}(undef, n_row, 1)
        new_distances = Array{Float64}(undef, n_row, 1)

        # TODO: Add `colwise` function (or use it from `Distances` package)
        # compute distances from the first centroid chosen to all the other data points
        first_centroid_matrix = convert(Matrix, centroids[1, :]')

        # flatten distances
        pairwise!(distances, X, first_centroid_matrix, mode)
        distances[rand_idx] = 0.0

        for i = 2:k
            # choose the next centroid, the probability for each data point to be chosen
            # is directly proportional to its squared distance from the nearest centroid
            r_idx = wsample(1:n_row, vec(distances))
            rand_indices[i] = r_idx
            centroids[i, :] .= X[r_idx, :]

            # no need for final distance update
            i == k && break

            # compute distances from the centroids to all data points
            current_centroid_matrix = convert(Matrix, centroids[i, :]')
            # new_distances = vec(pairwise(SqEuclidean(), X, current_centroid_matrix, dims = 1))
            pairwise!(new_distances, X, first_centroid_matrix, mode)

            # and update the squared distance as the minimum distance to all centroid
            for i in 1:n_row
                distances[i, 1] = distances[i, 1] < new_distances[i, 1] ? distances[i, 1] : new_distances[i, 1]
            end
            distances[r_idx, 1] = 0.0
        end

    else
        # randomly select points from the design matrix as the initial centroids
        rand_indices = rand(1:n_row, k)
        centroids = X[rand_indices, :]
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
            s += (x[i, j] - centre[labels[i], j])^2
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

    n_row, n_col = size(design_matrix)
    centroids = init == nothing ? smart_init(design_matrix, k, mode, init=k_init).centroids : init

    labels = Vector{Int}(undef, n_row)
    distances = Vector{Float64}(undef, n_row)
    centroids_cnt = Vector{Int}(undef, size(centroids, 1))

    J_previous = Inf64

    nearest_neighbour = Array{Float64, 2}(undef, size(design_matrix, 1), size(centroids, 1))
    # Update centroids & labels with closest members until convergence
    for iter = 1:max_iters
        pairwise!(nearest_neighbour, design_matrix, centroids, mode)

        @inbounds for i in axes(nearest_neighbour, 1)
            labels[i] = 1
            distances[i] = nearest_neighbour[i, 1]
            for j in 2:size(nearest_neighbour, 2)
                if distances[i] > nearest_neighbour[i, j]
                    labels[i] = j
                    distances[i] = nearest_neighbour[i, j]
                end
            end
        end

        centroids .= 0.0
        centroids_cnt .= 0
        @inbounds for i in axes(design_matrix, 1)
            centroids[labels[i], 1] += design_matrix[i, 1]
            centroids_cnt[labels[i]] += 1
        end
        @inbounds for j in 2:n_col
            for i in axes(design_matrix, 1)
                centroids[labels[i], j] += design_matrix[i, j]
            end
        end
        centroids ./= centroids_cnt

        # Cost objective
        J = mean(distances)

        if verbose
            # Show progress and terminate if J stopped decreasing.
            println("Iteration $iter: Jclust = $J.")
        end

        # Final Step: Check for convergence
        if (iter > 1) & (abs(J - J_previous) < (tol * J))

            sum_squares = sum_of_squares(design_matrix, labels, centroids)

            # Terminate algorithm with the assumption that K-means has converged
            if verbose
                println("Successfully terminated with convergence.")
            end

            return (labels=labels, centroids=centroids, sum_squares=sum_squares)

        elseif (iter == max_iters) & (abs(J - J_previous) > (tol * J))
            throw(error("Failed to converge Check data and/or implementation or increase max_iter."))

        end

        J_previous = J
    end
end

end # module
