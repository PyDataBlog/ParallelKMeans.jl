"""
    LightElkan <: AbstractKMeansAlg

Simplified version of Elkan algorithm for k-means calculation. This algorithm
gives the same results as basic Lloyd algorithm, but improve in speed by omitting
unnecessary calculations. In this implementation there are two conditions applied
to accelerate calculations

- if point is sufficiently close to it's centroid, i.e. distance to the centroid is smaller than
half minimum distance from centroid to all other centroid. In this scenario point immediately get
label of closest centroid.
- if during calculation of new label distance from the point to current centroid is less than
half of the distance from centroid to any other centroid, then distance from the point to
other centroid is not calculated.

One has to take into account, that LightElkan algorithm has an overhead of the calculation
k x k matrix of centroids distances, so for tasks with no apparent cluster structure may perform
worser than basic LLoyd algorithm.
"""
struct LightElkan <: AbstractKMeansAlg end

function kmeans!(alg::LightElkan, containers, X, k;
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = 1e-6, verbose = false, init = nothing)
    nrow, ncol = size(X)
    centroids = init == nothing ? smart_init(X, k, n_threads, init=k_init).centroids : deepcopy(init)

    converged = false
    niters = 1
    J_previous = 0.0

    # Update centroids & labels with closest members until convergence
    while niters <= max_iters
        update_containers(alg, containers, centroids, n_threads)
        @parallelize n_threads ncol chunk_update_centroids(alg, containers, centroids, X)
        collect_containers(alg, containers, centroids, n_threads)
        J = sum(containers.J)

        if verbose
            # Show progress and terminate if J stopped decreasing.
            println("Iteration $niters: Jclust = $J")
        end

        # Check for convergence
        if (niters > 1) & (abs(J - J_previous) < (tol * J))
            converged = true
            break
        end

        J_previous = J
        niters += 1
    end

    @parallelize n_threads ncol sum_of_squares(containers, X, containers.labels, centroids)
    totalcost = sum(containers.sum_of_squares)

    # Terminate algorithm with the assumption that K-means has converged
    if verbose & converged
        println("Successfully terminated with convergence.")
    end

    # TODO empty placeholder vectors should be calculated
    # TODO Float64 type definitions is too restrictive, should be relaxed
    # especially during GPU related development
    return KmeansResult(centroids, containers.labels, Float64[], Int[], Float64[], totalcost, niters, converged)
end

"""
    create_containers(::LightElkan, k, nrow, ncol, n_threads)

Internal function for the creation of all necessary intermidiate structures.

- `new_centroids` - container which holds new positions of centroids
- `centroids_cnt` - container which holds number of points for each centroid
- `labels` - vector which holds labels of corresponding points
- `centroids_dist` - symmetric matrix k x k which holds weighted distances between centroids
"""
function create_containers(alg::LightElkan, k, nrow, ncol, n_threads)
    lng = n_threads + 1
    centroids_new = Vector{Array{Float64,2}}(undef, lng)
    centroids_cnt = Vector{Vector{Int}}(undef, lng)

    for i in 1:lng
        centroids_new[i] = Array{Float64, 2}(undef, nrow, k)
        centroids_cnt[i] = Vector{Int}(undef, k)
    end

    labels = zeros(Int, ncol)

    J = Vector{Float64}(undef, n_threads)

    centroids_dist = Matrix{Float64}(undef, k, k)

    # total_sum_calculation
    sum_of_squares = Vector{Float64}(undef, n_threads)

    return (centroids_new = centroids_new, centroids_cnt = centroids_cnt,
            labels = labels, centroids_dist = centroids_dist,
            J = J, sum_of_squares = sum_of_squares)
end


"""
    update_containers!(::LightElkan, containers, centroids, n_threads)

Internal function for the `LightElkan` algorithm which updates distances
between centroids. These distances are presented as symmetric matrix,
on diagonal is written minimal distance from current centroid to all other.
All distances are weighted with the factor 0.25 in order to simplify following
update_centroids calculations.
"""
function update_containers(::LightElkan, containers, centroids, n_threads)
    # unpack containers for easier manipulations
    centroids_dist = containers.centroids_dist

    k = size(centroids_dist, 1) # number of clusters
    @inbounds for j in axes(centroids_dist, 2)
        min_dist = Inf
        for i in j + 1:k
            d = distance(centroids, centroids, i, j)
            centroids_dist[i, j] = d
            centroids_dist[j, i] = d
            min_dist = min_dist < d ? min_dist : d
        end
        for i in 1:j - 1
            min_dist = min_dist < centroids_dist[j, i] ? min_dist : centroids_dist[j, i]
        end
        centroids_dist[j, j] = min_dist
    end

    # TODO: oh, one should be careful here. inequality holds for eucledian metrics
    # not square eucledian. So, for Lp norm it should be something like
    # centroids_dist = 0.5^p. Should check one more time original paper
    centroids_dist .*= 0.25

    return centroids_dist
end

"""
    chunk_update_centroids!(centroids, containers, ::AbstractKMeansAlg, design_matrix, r, idx)

Internal function which calculates single centroids update for data chunk.

Argument `idx` denotes number of the thread used, if it is equals 0 it means, that we are in single
thread mode.
"""
function chunk_update_centroids(::LightElkan, containers, centroids,  X, r, idx)

    # unpack containers for easier manipulations
    centroids_new = containers.centroids_new[idx]
    centroids_cnt = containers.centroids_cnt[idx]
    centroids_dist = containers.centroids_dist
    labels = containers.labels

    centroids_new .= 0.0
    centroids_cnt .= 0
    J = 0.0
    @inbounds for i in r
        # calculate distance to the previous center
        label = labels[i] > 0 ? labels[i] : 1
        last_label = label
        dist = distance(X, centroids, i, label)

        min_distance = dist

        # we can optimize in two ways
        # if point is close (less then centroids_dist[i, i]) to the center then there is no need to recalculate it
        # if it's not close, then we can skip some of centers if the center is too far away from
        # current point (Elkan triangular inequality)
        if min_distance > centroids_dist[label, label]
            for k in axes(centroids, 2)
                k == last_label && continue
                # triangular inequality
                centroids_dist[k, label] > min_distance && continue
                dist = distance(X, centroids, i, k)
                label = min_distance > dist ? k : label
                min_distance = min_distance > dist ? dist : min_distance
            end
        end

        labels[i] = label
        centroids_cnt[label] += 1
        for j in axes(X, 1)
            centroids_new[j, label] += X[j, i]
        end
        J += min_distance
    end

    containers.J[idx] = J
end
