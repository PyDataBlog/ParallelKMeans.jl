"""
    Hamerly()

Hamerly algorithm implementation, based on "Hamerly, Greg. (2010). Making k-means Even Faster.
 Proceedings of the 2010 SIAM International Conference on Data Mining. 130-140. 10.1137/1.9781611972801.12."

This algorithm provides much faster convergence than Lloyd algorithm with realtively small increase in
memory footprint. It is especially suitable for low to medium dimensional input data.

It can be used directly in `kmeans` function

```julia
X = rand(30, 100_000)   # 100_000 random points in 30 dimensions

kmeans(Hamerly(), X, 3) # 3 clusters, Hamerly algorithm
```
"""
struct Hamerly <: AbstractKMeansAlg end


function kmeans!(alg::Hamerly, containers, X, k, weights=nothing, metric=Euclidean();
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = eltype(X)(1e-6), verbose = false,
                init = nothing, rng = Random.GLOBAL_RNG)

    nrow, ncol = size(X)
    centroids = init == nothing ? smart_init(X, k, n_threads, weights, rng, init=k_init).centroids : deepcopy(init)

    @parallelize n_threads ncol chunk_initialize(alg, containers, centroids, X, weights, metric)

    T = eltype(X)
    converged = false
    niters = 0
    J_previous = zero(T)
    p = containers.p

    # Update centroids & labels with closest members until convergence
    while niters < max_iters
        niters += 1
        update_containers(alg, containers, centroids, n_threads, metric)
        @parallelize n_threads ncol chunk_update_centroids(alg, containers, centroids, X, weights, metric)
        collect_containers(alg, containers, n_threads)

        J = sum(containers.ub)
        move_centers(alg, containers, centroids, metric)

        r1, r2, pr1, pr2 = double_argmax(p)
        @parallelize n_threads ncol chunk_update_bounds(alg, containers, r1, r2, pr1, pr2, metric)

        if verbose
            # Show progress and terminate if J stops decreasing as specified by the tolerance level.
            println("Iteration $niters: Jclust = $J")
        end

        # Check for convergence
        if (niters > 1) & (abs(J - J_previous) < (tol * J))
            converged = true
            break
        end

        J_previous = J
    end

    @parallelize n_threads ncol sum_of_squares(containers, X, containers.labels, centroids, weights, metric)
    totalcost = sum(containers.sum_of_squares)

    # Terminate algorithm with the assumption that K-means has converged
    if verbose & converged
        println("Successfully terminated with convergence.")
    end

    counts = collect(values(sort(countmap(containers.labels))))

    # TODO empty placeholder vectors should be calculated
    # TODO Float64 type definitions is too restrictive, should be relaxed
    # especially during GPU related development
    return KmeansResult(centroids, containers.labels, T[], counts, T[], totalcost, niters, converged)
end


function create_containers(alg::Hamerly, X, k, nrow, ncol, n_threads)
    T = eltype(X)
    lng = n_threads + 1
    centroids_new = Vector{Matrix{T}}(undef, lng)
    centroids_cnt = Vector{Vector{T}}(undef, lng)

    for i = 1:lng
        centroids_new[i] = zeros(T, nrow, k)
        centroids_cnt[i] = zeros(T, k)
    end

    # Upper bound to the closest center
    ub = Vector{T}(undef, ncol)

    # lower bound to the second closest center
    lb = Vector{T}(undef, ncol)

    labels = zeros(Int, ncol)

    # distance that centroid has moved
    p = Vector{T}(undef, k)

    # distance from the center to the closest other center
    s = Vector{T}(undef, k)

    # total_sum_calculation
    sum_of_squares = Vector{T}(undef, n_threads)

    return (
        centroids_new = centroids_new,
        centroids_cnt = centroids_cnt,
        labels = labels,
        ub = ub,
        lb = lb,
        p = p,
        s = s,
        sum_of_squares = sum_of_squares
    )
end


"""
    chunk_initialize(alg::Hamerly, containers, centroids, X, weights, metric, r, idx)

Initial calulation of all bounds and points labeling.
"""
function chunk_initialize(alg::Hamerly, containers, centroids, X, weights, metric, r, idx)
    T = eltype(X)
    centroids_cnt = containers.centroids_cnt[idx]
    centroids_new = containers.centroids_new[idx]

    @inbounds for i in r
        label = point_all_centers!(containers, centroids, X, i, metric)
        centroids_cnt[label] += isnothing(weights) ? one(T) : weights[i]
        for j in axes(X, 1)
            centroids_new[j, label] += isnothing(weights) ? X[j, i] : weights[i] * X[j, i]
        end
    end
end


"""
    update_containers(::Hamerly, containers, centroids, n_threads, metric)

Calculates minimum distances from centers to each other.
"""
function update_containers(::Hamerly, containers, centroids, n_threads, metric)
    T = eltype(centroids)
    s = containers.s
    s .= T(Inf)
    @inbounds for i in axes(centroids, 2)
        for j in i+1:size(centroids, 2)
            d = T(centers_coefficient(metric)) * distance(metric, centroids, centroids, i, j)
            s[i] = s[i] > d ? d : s[i]
            s[j] = s[j] > d ? d : s[j]
        end
    end
end


"""
    chunk_update_centroids(alg::Hamerly, containers, centroids, X, weights, metric, r, idx)

Detailed description of this function can be found in the original paper. It iterates through
all points and tries to skip some calculation using known upper and lower bounds of distances
from point to centers. If it fails to skip than it fall back to generic `point_all_centers!` function.
"""
function chunk_update_centroids(alg::Hamerly, containers, centroids, X, weights, metric, r, idx)

    # unpack containers for easier manipulations
    centroids_new = containers.centroids_new[idx]
    centroids_cnt = containers.centroids_cnt[idx]
    labels = containers.labels
    s = containers.s
    lb = containers.lb
    ub = containers.ub
    T = eltype(X)

    @inbounds for i in r
        # m ← max(s(a(i))/2, l(i))
        m = max(s[labels[i]], lb[i])
        # first bound test
        if ub[i] > m
            # tighten upper bound
            label = labels[i]
            ub[i] = distance(metric, X, centroids, i, label)
            # second bound test
            if ub[i] > m
                label_new = point_all_centers!(containers, centroids, X, i, metric)
                if label != label_new
                    labels[i] = label_new
                    centroids_cnt[label_new] += isnothing(weights) ? one(T) : weights[i]
                    centroids_cnt[label] -= isnothing(weights) ? one(T) : weights[i]
                    for j in axes(X, 1)
                        centroids_new[j, label_new] += isnothing(weights) ? X[j, i] : weights[i] * X[j, i]
                        centroids_new[j, label] -= isnothing(weights) ? X[j, i] : weights[i] * X[j, i]
                    end
                end
            end
        end
    end
end


"""
    point_all_centers!(containers, centroids, X, i, metric)

Calculates new labels and upper and lower bounds for all points.
"""
function point_all_centers!(containers, centroids, X, i, metric)
    ub = containers.ub
    lb = containers.lb
    labels = containers.labels
    T = eltype(X)

    min_distance = T(Inf)
    min_distance2 = T(Inf)
    label = 1
    @inbounds for k in axes(centroids, 2)
        dist = distance(metric, X, centroids, i, k)
        if min_distance > dist
            label = k
            min_distance2 = min_distance
            min_distance = dist
        elseif min_distance2 > dist
            min_distance2 = dist
        end
    end

    ub[i] = min_distance
    lb[i] = min_distance2
    labels[i] = label

    return label
end


"""
    move_centers(::Hamerly, containers, centroids, metric)

Calculates new positions of centers and distance they have moved. Results are stored
in `centroids` and `p` respectively.
"""
function move_centers(::Hamerly, containers, centroids, metric)
    centroids_new = containers.centroids_new[end]
    p = containers.p
    T = eltype(centroids)

    @inbounds for i in axes(centroids, 2)
        d = distance(metric, centroids, centroids_new, i, i)
        for j in axes(centroids, 1)
            centroids[j, i] = centroids_new[j, i]
        end
        p[i] = d
    end
end


"""
    chunk_update_bounds(alg::Hamerly, containers, r1, r2, pr1, pr2, metric::Euclidean, r, idx)

Updates upper and lower bounds of point distance to the centers, with regard to the centers movement
when metric is Euclidean.
"""
function chunk_update_bounds(alg::Hamerly, containers, r1, r2, pr1, pr2, metric::Euclidean, r, idx)
    p = containers.p
    ub = containers.ub
    lb = containers.lb
    labels = containers.labels
    T = eltype(containers.ub)

    # Since bounds are squared distance, `sqrt` is used to make corresponding estimation, unlike
    # the original paper, where usual metric is used.
    #
    # Using notation from original paper, `u` is upper bound and `a` is `labels`, so
    #
    # `u[i] -> u[i] + p[a[i]]`
    #
    # then squared distance is
    #
    # `u[i]^2 -> (u[i] + p[a[i]])^2 = u[i]^2 + 2 p[a[i]] u[i] + p[a[i]]^2`
    #
    # Taking into account that in our noations `p^2 -> p`, `u^2 -> ub` we obtain
    #
    # `ub[i] -> ub[i] + 2 sqrt(p[a[i]] ub[i]) + p[a[i]]`
    #
    # The same applies to the lower bounds.
    @inbounds for i in r
        label = labels[i]
        ub[i] += T(2)*sqrt(abs(ub[i] * p[label])) + p[label]
        if r1 == label
            lb[i] = lb[i] <= pr2 ? zero(T) : lb[i] + pr2 - T(2)*sqrt(abs(pr2*lb[i]))
        else
            lb[i] = lb[i] <= pr1 ? zero(T) : lb[i] + pr1 - T(2)*sqrt(abs(pr1*lb[i]))
        end
    end
end


"""
    chunk_update_bounds(alg::Hamerly, containers, r1, r2, pr1, pr2, metric::Metric, r, idx)

Updates upper and lower bounds of point distance to the centers, with regard to the centers movement
when metric is Euclidean.
"""
function chunk_update_bounds(alg::Hamerly, containers, r1, r2, pr1, pr2, metric::Metric, r, idx)
    p = containers.p
    ub = containers.ub
    lb = containers.lb
    labels = containers.labels
    T = eltype(containers.ub)
    # Using notation from original paper, `u` is upper bound and `a` is `labels`, so
    # `u[i] -> u[i] + p[a[i]]`
    @inbounds for i in r
        label = labels[i]
        ub[i] += p[label]
        lb[i] -= r1 == label ? pr2 : pr1
    end
end


"""
    double_argmax(p)

Finds maximum and next after maximum arguments.
"""
function double_argmax(p::AbstractVector{T}) where T
    r1, r2 = 1, 1
    d1 = p[1]
    d2 = T(-Inf)
    for i in 2:length(p)
        if p[i] > d1
            r2 = r1
            r1 = i
            d2 = d1
            d1 = p[i]
        elseif p[i] > d2
            d2 = p[i]
            r2 = i
        end
    end

    r1, r2, d1, d2
end
