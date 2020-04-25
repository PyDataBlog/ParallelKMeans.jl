"""
    spliiter(n, k)

Internal utility function, splits 1:n sequence to k chunks of approximately same size.
"""
function splitter(n, k)
    xz = Int.(ceil.(range(0, n, length = k+1)))
    return [xz[i]+1:xz[i+1] for i in 1:k]
end

"""
    colwise!(target, x, y, n_threads)

Internal function for colwise calculations. Let `x` is a matrix `m x n` and `y` is a vector of the length `m`.
Then the `colwise!` function computes distance between each column in `x` and `y` and store result
in `target` array. Argument `n_threads` defines the number of threads used for calculation.
"""
function colwise!(target, x, y, n_threads = Threads.nthreads())
    ncol = size(x, 2)

    # we could have used same algorithm for n_threads == 1
    # but ran into unnecessary allocations and split calculations.
    # Impact is neglible, yet there is no need to do extra calculations
    if n_threads != 1
        ranges = splitter(ncol, n_threads)
        waiting_list = Task[]

        for i in 1:length(ranges) - 1
            push!(waiting_list, @spawn chunk_colwise!(target, x, y, ranges[i]))
        end
        chunk_colwise!(target, x, y, ranges[end])

        for i in 1:length(ranges) - 1
            wait(waiting_list[i])
        end
    else
        chunk_colwise!(target, x, y, axes(x, 2))
    end

    target
end

"""
    chunk_colwise!(target, x, y, r)

Utility function for calculation of the `colwise!(target, x, y, n_threads)` function.
UnitRange argument `r` select subarray of original design matrix `x` that is going
to be processed.
"""
function chunk_colwise!(target, x, y, r)
    T = eltype(x)
    @inbounds for j in r
        res = zero(T)
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

A named tuple representing centroids and indices respecitively is returned.
"""
function smart_init(X, k, n_threads = Threads.nthreads();
        init = "k-means++")

    n_row, n_col = size(X)
    T = eltype(X)

    if init == "k-means++"

        # randonmly select the first centroid from the data (X)

        # TODO relax constraints on distances, may be should
        # define `X` as X::AbstractArray{T} where {T <: Number}
        # and use this T for all calculations.
        centroids = zeros(T, n_row, k)
        rand_indices = Vector{Int}(undef, k)
        rand_idx = rand(1:n_col)
        rand_indices[1] = rand_idx
        centroids[:, 1] .= X[:, rand_idx]
        distances = Vector{T}(undef, n_col)
        new_distances = Vector{T}(undef, n_col)

        # compute distances from the first centroid chosen to all the other data points
        colwise!(distances, X, centroids[:, 1], n_threads)
        distances[rand_idx] = zero(T)

        for i = 2:k
            # choose the next centroid, the probability for each data point to be chosen
            # is directly proportional to its squared distance from the nearest centroid
            r_idx = wsample(1:n_col, vec(distances))
            rand_indices[i] = r_idx
            centroids[:, i] .= X[:, r_idx]

            # no need for final distance update
            i == k && break

            # compute distances from the centroids to all data points
            colwise!(new_distances, X, centroids[:, i], n_threads)

            # and update the squared distance as the minimum distance to all centroid
            for i in 1:n_col
                distances[i] = distances[i] < new_distances[i] ? distances[i] : new_distances[i]
            end
            distances[r_idx] = zero(T)
        end

    else
        # randomly select points from the design matrix as the initial centroids
        rand_indices = sample(1:n_col, k, replace = false)
        centroids = X[:, rand_indices]
    end

    return (centroids = centroids, indices = rand_indices)
end
