"""
    spliiter(n, k)

Internal utility function, splits 1:n sequence to k chunks of approximately same size.
"""
function splitter(n, k)
    xz = Int.(ceil.(range(0, n, length = k+1)))
    return [xz[i]+1:xz[i+1] for i in 1:k]
end


"""
    chunk_colwise!(target, x, y, i, weights, r, idx)

Utility function for the calculation of the weighted distance between points `x` and
centroid vector `y[:, i]`.
UnitRange argument `r` select subarray of original design matrix `x` that is going
to be processed.
"""
function chunk_colwise(target, x, y, i, weights, r, idx)
    T = eltype(x)
    @inbounds for j in r
        dist = distance(x, y, j, i)
        dist = isnothing(weights) ? dist : weights[j] * dist
        target[j] = dist < target[j] ? dist : target[j]
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
function smart_init(X, k, n_threads = Threads.nthreads(), weights = nothing;
        init = "k-means++")

    nrow, ncol = size(X)
    T = eltype(X)
    centroids = Matrix{T}(undef, nrow, k)
    rand_indices = Vector{Int}(undef, k)

    if init == "k-means++"

        # randonmly select the first centroid from the data (X)

        # TODO relax constraints on distances, may be should
        # define `X` as X::AbstractArray{T} where {T <: Number}
        # and use this T for all calculations.
        rand_idx = isnothing(weights) ? rand(1:ncol) : wsample(1:ncol, weights)
        rand_indices[1] = rand_idx
        @inbounds for j in axes(X, 1)
            centroids[j, 1] = X[j, rand_idx]
        end
        # centroids[:, 1] .= @view X[:, rand_idx]
        # distances = Vector{T}(undef, ncol)
        # new_distances = Vector{T}(undef, ncol)
        distances = fill(T(Inf), ncol)

        # compute distances from the first centroid chosen to all the other data points
        @parallelize n_threads ncol chunk_colwise(distances, X, centroids, 1, weights)
        distances[rand_idx] = zero(T)

        for i = 2:k
            # choose the next centroid, the probability for each data point to be chosen
            # is directly proportional to its squared distance from the nearest centroid
            r_idx = wsample(1:ncol, distances)
            rand_indices[i] = r_idx
            @inbounds for j in axes(X, 1)
                centroids[j, i] = X[j, r_idx]
            end

            # no need for final distance update
            i == k && break

            # compute distances from the centroids to all data points
            @parallelize n_threads ncol chunk_colwise(distances, X, centroids, i, weights)

            distances[r_idx] = zero(T)
        end

    else
        # randomly select points from the design matrix as the initial centroids
        rand_indices .= sample(1:ncol, k, replace = false)
        centroids .= @view X[:, rand_indices]
    end

    return (centroids = centroids, indices = rand_indices)
end
