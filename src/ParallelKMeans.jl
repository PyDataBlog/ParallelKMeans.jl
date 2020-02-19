module ParallelKMeans


using StatsBase
import Base.Threads: @spawn, @threads

export kmeans

"""
TODO 1: Document function
"""
function divider(n, k)
    d = div(n, k)
    xz = vcat(collect((0:k-1) * d), n)
    return [t[1]:t[2] for t in zip(xz[1:end-1] .+ 1, xz[2:end])]
end


"""
TODO 2: Document function
"""
function pl_pairwise!(target, x, y, nth = Threads.nthreads())
    ncol = size(x, 2)
    nrow = size(x, 1)
    ranges = divider(nrow, nth)
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
TODO 3: Document function
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
TODO 4: Document function
"""
function pairwise!(target, x, y)
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
    smart_init(X, k; init="k-means++")

    This function handles the random initialisation of the centroids from the
    design matrix (X) and desired groups (k) that a user supplies.

    `k-means++` algorithm is used by default with the normal random selection
    of centroids from X used if any other string is attempted.

    A tuple representing the centroids, number of rows, & columns respecitively
    is returned.
"""
function smart_init(X::Array{Float64, 2}, k::Int; init::String="k-means++")
    n_row, n_col = size(X)

    if init == "k-means++"

        # randonmly select the first centroid from the data (X)
        centroids = zeros(k, n_col)
        rand_idx = rand(1:n_row)
        centroids[1, :] .= X[rand_idx, :]
        distances = Array{Float64}(undef, n_row, 1)
        new_distances = Array{Float64}(undef, n_row, 1)

        # compute distances from the first centroid chosen to all the other data points
        first_centroid_matrix = convert(Matrix, centroids[1, :]')

        # flatten distances
        # distances = vec(pairwise(SqEuclidean(), X, first_centroid_matrix, dims = 1))
        pairwise!(distances, X, first_centroid_matrix)

        for i = 2:k
            # choose the next centroid, the probability for each data point to be chosen
            # is directly proportional to its squared distance from the nearest centroid
            r_idx = sample(1:n_row, ProbabilityWeights(vec(distances)))
            centroids[i, :] .= X[r_idx, :]

            # Ignore setting the last centroid to help the separation of centroids
            if i == (k-1)
                break
            end

            # compute distances from the centroids to all data points
            current_centroid_matrix = convert(Matrix, centroids[i, :]')
            # new_distances = vec(pairwise(SqEuclidean(), X, current_centroid_matrix, dims = 1))
            pairwise!(new_distances, X, first_centroid_matrix)

            # and update the squared distance as the minimum distance to all centroid
            # distances = minimum([distances, new_distances])
            for i in 1:n_row
                distances[i, 1] = distances[i, 1] < new_distances[i, 1] ? distances[i, 1] : new_distances[i, 1]
            end
        end

    else
        # randomly select points from the design matrix as the initial centroids
        rand_indices = rand(1:n_row, k)
        centroids = X[rand_indices, :]

    end

    return centroids, n_row, n_col
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
function kmeans(design_matrix::Array{Float64, 2}, k::Int; k_init::String = "k-means++",
    max_iters::Int = 300, tol = 1e-4, verbose::Bool = true)

    centroids, n_row, n_col = smart_init(design_matrix, k, init=k_init)

    labels = Vector{Int}(undef, n_row)
    distances = Vector{Float64}(undef, n_row)
    centroids_cnt = Vector{Int}(undef, size(centroids, 1))

    J_previous = Inf64

    nearest_neighbour = Array{Float64, 2}(undef, size(design_matrix, 1), size(centroids, 1))
    # Update centroids & labels with closest members until convergence
    for iter = 1:max_iters
        pairwise!(nearest_neighbour, design_matrix, centroids)

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
            println("Iteration ", iter, ": Jclust = ", J, ".")
        end

        # Final Step: Check for convergence
        if iter > 1 && abs(J - J_previous) < (tol * J)

            sum_squares = sum_of_squares(design_matrix, labels, centroids)

            # Terminate algorithm with the assumption that K-means has converged
            if verbose
                println("Successfully terminated with convergence.")
            end

            return labels, centroids, sum_squares

        elseif iter == max_iters && abs(J - J_previous) > (tol * J)
            throw(error("Failed to converge Check data and/or implementation or increase max_iter."))

        end

        J_previous = J
    end

end

end # module
