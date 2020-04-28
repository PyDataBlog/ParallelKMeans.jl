"""
    Yinyang()

Yinyang algorithm implementation, based on "Yufei Ding et al. 2015. Yinyang K-Means: A Drop-In
Replacement of the Classic K-Means with Consistent Speedup. Proceedings of the 32nd International
Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015"

Generally it outperform `Hamerly` algorithm and has roughly the same time as `Elkan`
algorithm with much lower memory consumption.


`Yinyang` supports following arguments:
`auto`: `Bool`, indicates whether to perform automated or manual grouping
`group_size`: `Int`, estimation of average number of clusters per group. Lower numbers
corresponds to higher calculation speed and higher memory consumption and vice versa.

It can be used directly in `kmeans` function

```julia
X = rand(30, 100_000)   # 100_000 random points in 30 dimensions

# 3 clusters, Yinyang algorithm, with deault 7 group_size
kmeans(Yinyang(), X, 3)

# Following are equivalent
# 3 clusters, Yinyang algorithm with 10 group_size
kmeans(Yinyang(group_size = 10), X, 3)
kmeans(Yinyang(10), X, 3)

# One group with the size of the number of points
kmeans(Yinyang(auto = false), X, 3)
kmeans(Yinyang(false), X, 3)

# Chinese writing can be used
kmeans(阴阳(), X, 3)
```
"""
struct Yinyang <: AbstractKMeansAlg
    auto::Bool
    group_size::Int
end

Yinyang(auto::Bool) = Yinyang(auto, 7)
Yinyang(group_size::Int) = Yinyang(true, group_size)
Yinyang(; group_size = 7, auto = true) = Yinyang(auto, group_size)
阴阳(auto::Bool) = Yinyang(auto, 7)
阴阳(group_size::Int) = Yinyang(true, group_size)
阴阳(; group_size = 7, auto = true) = Yinyang(auto, group_size)

function kmeans!(alg::Yinyang, containers, X, k, weights;
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = 1e-6, verbose = false,
                init = nothing, rng = Random.GLOBAL_RNG)
    nrow, ncol = size(X)
    centroids = init == nothing ? smart_init(X, k, n_threads, weights, rng, init=k_init).centroids : deepcopy(init)

    # create initial groups of centers, step 1 in original paper
    initialize(alg, containers, centroids, rng, n_threads)
    # construct initial bounds, step 2
    @parallelize n_threads ncol chunk_initialize(alg, containers, centroids, X, weights)
    collect_containers(alg, containers, n_threads)

    # update centers and calculate drifts. Step 3.1 of the original paper.
    calculate_centroids_movement(alg, containers, centroids)

    T = eltype(X)
    converged = false
    niters = 0
    J_previous = zero(T)

    # Update centroids & labels with closest members until convergence
    while niters < max_iters
        niters += 1
        J = sum(containers.ub)
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

        # push!(containers.debug, [0, 0, 0])
        # Core calculation of the Yinyang, 3.2-3.3 steps of the original paper
        @parallelize n_threads ncol chunk_update_centroids(alg, containers, centroids, X, weights)
        collect_containers(alg, containers, n_threads)

        # update centers and calculate drifts. Step 3.1 of the original paper.
        calculate_centroids_movement(alg, containers, centroids)
    end

    @parallelize n_threads ncol sum_of_squares(containers, X, containers.labels, centroids, weights)
    totalcost = sum(containers.sum_of_squares)

    # Terminate algorithm with the assumption that K-means has converged
    if verbose & converged
        println("Successfully terminated with convergence.")
    end

    # TODO empty placeholder vectors should be calculated
    # TODO Float64 type definitions is too restrictive, should be relaxed
    # especially during GPU related development
    return KmeansResult(centroids, containers.labels, T[], Int[], T[], totalcost, niters, converged)
end

function create_containers(alg::Yinyang, X, k, nrow, ncol, n_threads)
    T = eltype(X)
    lng = n_threads + 1
    centroids_new = Vector{Matrix{T}}(undef, lng)
    centroids_cnt = Vector{Vector{T}}(undef, lng)

    for i = 1:lng
        centroids_new[i] = zeros(T, nrow, k)
        centroids_cnt[i] = zeros(T, k)
    end

    mask = Vector{Vector{Bool}}(undef, n_threads)
    for i in 1:n_threads
        mask[i] = Vector{Bool}(undef, k)
    end

    if alg.auto
        t = k ÷ alg.group_size
        t = t < 1 ? 1 : t
    else
        t = 1
    end

    labels = zeros(Int, ncol)

    ub = Vector{T}(undef, ncol)

    lb = Matrix{T}(undef, t, ncol)

    # maximum group drifts
    gd = Vector{T}(undef, t)

    # distance that centroid has moved
    p = Vector{T}(undef, k)

    # Group indices
    groups = Vector{UnitRange{Int}}(undef, t)

    # mapping between cluster center and group
    indices = Vector{Int}(undef, k)

    # total_sum_calculation
    sum_of_squares = Vector{T}(undef, n_threads)

    # debug = []

    return (
        centroids_new = centroids_new,
        centroids_cnt = centroids_cnt,
        labels = labels,
        sum_of_squares = sum_of_squares,
        p = p,
        ub = ub,
        lb = lb,
        groups = groups,
        indices = indices,
        gd = gd,
        mask = mask,
        # debug = debug
    )
end

function initialize(alg::Yinyang, containers, centroids, rng, n_threads)
    groups = containers.groups
    indices = containers.indices
    if length(groups) == 1
        groups[1] = axes(centroids, 2)
        indices .= 1
    else
        init_clusters = kmeans(Lloyd(), centroids, length(groups),
                                max_iters = 5, tol = 1e-10,
                                verbose = false, rng = rng)
        perm = sortperm(init_clusters.assignments)
        indices .= init_clusters.assignments[perm]
        groups .= rangify(indices)
    end
end

function chunk_initialize(alg::Yinyang, containers, centroids, X, weights, r, idx)
    T = eltype(X)
    centroids_cnt = containers.centroids_cnt[idx]
    centroids_new = containers.centroids_new[idx]

    @inbounds for i in r
        label = point_all_centers!(alg, containers, centroids, X, i)
        centroids_cnt[label] += isnothing(weights) ? one(T) : weights[i]
        for j in axes(X, 1)
            centroids_new[j, label] += isnothing(weights) ? X[j, i] : weights[i] * X[j, i]
        end
    end
end

function calculate_centroids_movement(alg::Yinyang, containers, centroids)
    p = containers.p
    groups = containers.groups
    gd = containers.gd
    centroids_new = containers.centroids_new[end]
    T = eltype(centroids)

    @inbounds for (gi, ri) in enumerate(groups)
        max_drift = T(-Inf)
        for i in ri
            p[i] = sqrt(distance(centroids, centroids_new, i, i))
            max_drift = p[i] > max_drift ? p[i] : max_drift

            # Should do it more elegantly
            for j in axes(centroids, 1)
                centroids[j, i] = centroids_new[j, i]
            end
        end
        gd[gi] = max_drift
    end
end

function chunk_update_centroids(alg::Yinyang, containers, centroids, X, weights, r, idx)
    # unpack containers for easier manipulations
    centroids_new = containers.centroids_new[idx]
    centroids_cnt = containers.centroids_cnt[idx]
    mask = containers.mask[idx]
    labels = containers.labels
    p = containers.p
    lb = containers.lb
    ub = containers.ub
    gd = containers.gd
    groups = containers.groups
    indices = containers.indices
    t = length(groups)
    T = eltype(X)

    @inbounds for i in r
        # update bounds
        # TODO: remove comment after becnhmarking
        # update_bounds(alg, ub, lb, labels, p, groups, gd, i)

        ub[i] += p[labels[i]]
        ubx = ub[i]
        lbx = T(Inf)
        for gi in 1:length(groups)
            lb[gi, i] -= gd[gi]
            lbx = lb[gi, i] < lbx ? lb[gi, i] : lbx
        end

        # Global filtering
        ubx <= lbx && continue
        # containers.debug[end][1] += 1 # number of misses

        # tighten upper bound
        label = labels[i]
        ubx = sqrt(distance(X, centroids, i, label))
        ub[i] = ubx
        ubx <= lbx && continue

        # local filter group which contains current label
        mask .= false
        ubx2 = ubx^2
        orig_group_id = indices[label]
        new_lb = lb[orig_group_id, i]
        old_label = label
        if ubx >= new_lb
            mask[old_label] = true
            ri = groups[orig_group_id]
            old_lb = new_lb + gd[orig_group_id] # recovering initial value of lower bound
            new_lb2 = T(Inf)
            for c in ri
                ((c == old_label) | (ubx < old_lb - p[c])) && continue
                mask[c] = true
                # containers.debug[end][2] += 1 # local filter update
                dist = distance(X, centroids, i, c)
                if dist < ubx2
                    new_lb2 = ubx2
                    ubx2 = dist
                    ubx = sqrt(dist)
                    label = c
                elseif dist < new_lb2
                    new_lb2 = dist
                end
            end
            new_lb = sqrt(new_lb2)
            for c in ri
                mask[c] && continue
                new_lb < old_lb - p[c] && continue
                # containers.debug[end][3] += 1 # lower bound update
                dist = distance(X, centroids, i, c)
                if dist < new_lb2
                    new_lb2 = dist
                    new_lb = sqrt(new_lb2)
                end
            end
            lb[orig_group_id, i] = new_lb
        end

        # Group filtering, now we know that previous best estimate of lower
        # bound was already claculated
        for gi in 1:t
            gi == orig_group_id && continue
            # Group filtering
            ubx < lb[gi, i] && continue
            new_lb = lb[gi, i]
            old_lb = new_lb + gd[gi]
            new_lb2 = T(Inf)
            ri = groups[gi]
            for c in ri
                # local filtering
                ubx < old_lb - p[c] && continue
                # containers.debug[end][2] += 1 # local filter update
                mask[c] = true
                dist = distance(X, centroids, i, c)
                if dist < ubx2
                    # closest center was in previous cluster
                    if indices[label] != gi
                        lb[indices[label], i] = ubx
                    else
                        new_lb = ubx
                    end
                    new_lb2 = ubx2
                    ubx2 = dist
                    ubx = sqrt(dist)
                    label = c
                elseif dist < new_lb2
                    new_lb2 = dist
                end
            end

            new_lb = sqrt(new_lb2)
            for c in ri
                mask[c] && continue
                new_lb < old_lb - p[c] && continue
                # containers.debug[end][3] += 1 # lower bound update
                dist = distance(X, centroids, i, c)
                if dist < new_lb2
                    new_lb2 = dist
                    new_lb = sqrt(new_lb2)
                end
            end

            lb[gi, i] = new_lb
        end

        # Assignment
        ub[i] = ubx
        if old_label != label
            labels[i] = label
            centroids_cnt[label] += isnothing(weights) ? one(T) : weights[i]
            centroids_cnt[old_label] -= isnothing(weights) ? one(T) : weights[i]
            for j in axes(X, 1)
                centroids_new[j, label] += isnothing(weights) ? X[j, i] : weights[i] * X[j, i]
                centroids_new[j, old_label] -= isnothing(weights) ? X[j, i] : weights[i] * X[j, i]
            end
        end
    end
end

"""
    point_all_centers!(containers, centroids, X, i)

Calculates new labels and upper and lower bounds for all points.
"""
function point_all_centers!(alg::Yinyang, containers, centroids, X, i)
    ub = containers.ub
    lb = containers.lb
    labels = containers.labels
    groups = containers.groups
    T = eltype(X)

    label = 1
    label2 = 1
    group_id = 1
    min_distance = T(Inf)
    @inbounds for (gi, ri) in enumerate(groups)
        group_min_distance = T(Inf)
        group_min_distance2 = T(Inf)
        group_label = ri[1]
        for k in ri
            dist = distance(X, centroids, i, k)
            if group_min_distance > dist
                group_label = k
                group_min_distance2 = group_min_distance
                group_min_distance = dist
            elseif group_min_distance2 > dist
                group_min_distance2 = dist
            end
        end
        if group_min_distance < min_distance
            lb[group_id, i] = sqrt(min_distance)
            lb[gi, i] = sqrt(group_min_distance2)
            group_id = gi
            min_distance = group_min_distance
            label = group_label
        else
            lb[gi, i] = sqrt(group_min_distance)
        end
    end

    ub[i] = sqrt(min_distance)
    labels[i] = label

    return label
end

# I believe there should be oneliner for it
function rangify(x)
    res = UnitRange{Int}[]
    id = 1
    val = x[1]
    for i in 2:length(x)
        if x[i] != val
            push!(res, id:i-1)
            id = i
            val = x[i]
        end
    end
    push!(res, id:length(x))

    return res
end

## Misc
# Borrowed from https://github.com/JuliaLang/julia/pull/21598/files

# swap columns i and j of a, in-place
# function swapcols!(a, i, j)
#     i == j && return
#     for k in axes(a,1)
#         @inbounds a[k,i], a[k,j] = a[k,j], a[k,i]
#     end
# end
#
# # like permute!! applied to each row of a, in-place in a (overwriting p).
# function permutecols!!(a, p)
#     count = 0
#     start = 0
#     while count < length(p)
#         ptr = start = findnext(!iszero, p, start+1)::Int
#         next = p[start]
#         count += 1
#         while next != start
#             swapcols!(a, ptr, next)
#             p[ptr] = 0
#             ptr = next
#             next = p[next]
#             count += 1
#         end
#         p[ptr] = 0
#     end
#     a
# end
