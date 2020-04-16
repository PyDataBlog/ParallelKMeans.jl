"""
    YingYang()

YingYang algorithm implementation, based on "Yufei Ding et al. 2015. Yinyang K-Means: A Drop-In
Replacement of the Classic K-Means with Consistent Speedup. Proceedings of the 32nd International
Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015"

It can be used directly in `kmeans` function

```julia
X = rand(30, 100_000)   # 100_000 random points in 30 dimensions

kmeans(YingYang(), X, 3) # 3 clusters, YingYang algorithm
```
"""
struct YingYang <: AbstractKMeansAlg
    auto::Bool
    divider::Int
end

YingYang() = YingYang(true, 10)
YingYang(auto::Bool) = YingYang(auto, 10)
YingYang(divider::Int) = YingYang(true, divider)

function kmeans!(alg::YingYang, containers, X, k;
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = 1e-6, verbose = false, init = nothing)
    nrow, ncol = size(X)
    centroids = init == nothing ? smart_init(X, k, n_threads, init=k_init).centroids : deepcopy(init)

    # create initial groups of centers, step 1 in original paper
    initialize(alg, containers, centroids)
    # construct initial bounds, step 2
    @parallelize n_threads ncol chunk_initialize(alg, containers, centroids, X)
    collect_containers(alg, containers, n_threads)

    converged = false
    niters = 0
    J_previous = 0.0

    # Update centroids & labels with closest members until convergence
    while niters < max_iters
        niters += 1
        # update centers and calculate drifts. Step 3.1 of the original paper.
        calculate_centroids_movement(alg, containers, centroids)

        J = sum(containers.ub)

        # Core calculation of the YingYang, 3.2-3.3 steps of the original paper
        @parallelize n_threads ncol chunk_update_centroids(alg, containers, centroids, X)

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

function create_containers(alg::YingYang, k, nrow, ncol, n_threads)
    lng = n_threads + 1
    centroids_new = Vector{Array{Float64,2}}(undef, lng)
    centroids_cnt = Vector{Vector{Int}}(undef, lng)

    for i = 1:lng
        centroids_new[i] = zeros(nrow, k)
        centroids_cnt[i] = zeros(k)
    end

    if alg.auto
        t = k ÷ alg.divider
        t = t < 1 ? 1 : t
    else
        t = 1
    end

    labels = zeros(Int, ncol)

    ub = Vector{Float64}(undef, ncol)

    lb = Matrix{Float64}(undef, t, ncol)

    # maximum group drifts
    gd = Vector{Float64}(undef, t)

    # distance that centroid has moved
    p = Vector{Float64}(undef, k)

    # Group indices
    groups = Vector{UnitRange{Int64}}(undef, t)

    # total_sum_calculation
    sum_of_squares = Vector{Float64}(undef, n_threads)

    return (
        centroids_new = centroids_new,
        centroids_cnt = centroids_cnt,
        labels = labels,
        sum_of_squares = sum_of_squares,
        p = p,
        ub = ub,
        lb = lb,
        groups = groups,
        gd = gd
    )
end

function chunk_initialize(alg::YingYang, containers, centroids, X, r, idx)
    centroids_cnt = containers.centroids_cnt[idx]
    centroids_new = containers.centroids_new[idx]

    @inbounds for i in r
        label = point_all_centers!(alg, containers, centroids, X, i)
        centroids_cnt[label] += 1
        for j in axes(X, 1)
            centroids_new[j, label] += X[j, i]
        end
    end
end

function calculate_centroids_movement(alg::YingYang, containers, centroids)
    p = containers.p
    groups = containers.groups
    gd = containers.gd
    centroids_new = containers.centroids_new[end]

    @inbounds for (gi, ri) in enumerate(groups)
        max_drift = -1.0
        for i in ri
            p[i] = distance(centroids, centroids_new, i, i)
            max_drift = p[i] > max_drift ? p[i] : max_drift
        end
        gd[gi] = max_drift
    end
end

function chunk_update_centroids(alg, containers, centroids, X, r, idx)
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

        @inbounds for i in r
            # update bounds
            # TODO: remove comment after becnhmarking
            # update_bounds(alg, ub, lb, labels, p, groups, gd, i)

            ub[i] += p[labels[i]]
            ubx = ub[i]
            lbx = Inf
            for gi in 1:length(groups)
                lb[gi, i] -= gd[gi]
                lbx = lb[gi, i] < lbx ? lb[g, i] : lbx
            end

            # Global filtering
            ubx <= lbx && continue

            # tighten upper bound
            label = labels[i]
            ubx = sqrt(distance(X, centroids, i, label))
            ub[i] = ubx
            ubx <= lbx && continue

            # local filter group which contains current label
            ubx2 = ubx^2
            old_label = label
            orig_group_id = indices[label]
            ri = groups[orig_group_id]
            new_lb = lb[orig_group_id, i]
            old_lb = new_lb + gd[orig_group_id]
            mask .= false
            mask[old_label] = true
            for c in ri
                ((c == old_label) | (ubx < old_lb - p[c])) && continue
                mask[c] = true
                dist = distance(X, centroids, i, c)
                if dist < ubx2
                    new_lb = ubx
                    ubx2 = dist
                    ubx = sqrt(dist)
                    label = c
                end
            end
            new_lb2 = new_lb^2
            for c in ri
                mask[c] && continue
                new_lb < old_lb - p[c] && continue
                dist = distance(X, centroids, i, c)
                if dist < new_lb2
                    new_lb2 = dist
                    new_lb = sqrt(new_lb2)
                end
            end
            lb[orig_group_id, i] = new_lb

            for gi in 1:t
                gi == orig_group_id && continue
                # Group filtering
                ubx <= lb[gi, i] && continue
                old_lb = lb[gi, i] + gd[di]
                ri = groups[gi]
                for c in ri
                    # local filtering
                    ubx < old_lb - p[c] && continue
                    mask[c] = true
                    dist = distance(X, centroids, i, c)
                    if dist < ubx2
                        # closest canter was in previous cluster
                        if indices[label] != gi
                            lb[indice[label]] = ubx
                        end
                        ubx2 = dist
                        ubx = sqrt(dist)

                        label = c

                    end
                end
                lb[gi, i] = lbg
            end
            ub[i] = ubx
            labels[i] = label

            # m ← max(s(a(i))/2, l(i))
            m = max(s[labels[i]], lb[i])
            # first bound test
            if ub[i] > m
                # tighten upper bound
                label = labels[i]
                ub[i] = distance(X, centroids, i, label)
                # second bound test
                if ub[i] > m
                    label_new = point_all_centers!(containers, centroids, X, i)
                    if label != label_new
                        labels[i] = label_new
                        centroids_cnt[label_new] += 1
                        centroids_cnt[label] -= 1
                        for j in axes(X, 1)
                            centroids_new[j, label_new] += X[j, i]
                            centroids_new[j, label] -= X[j, i]
                        end
                    end
                end
            end
        end
end

"""
    point_all_centers!(containers, centroids, X, i)

Calculates new labels and upper and lower bounds for all points.
"""
function point_all_centers!(alg::YingYang, containers, centroids, X, i)
    ub = containers.ub
    lb = containers.lb
    labels = containers.labels
    labels2 = containers.labels2
    groups = containers.groups

    label = 1
    label2 = 1
    group_id = 1
    min_distance = Inf
    @inbounds for (gi, ri) in enumerate(groups)
        group_min_distance = Inf
        group_min_distance2 = Inf
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
            lb[group_id, i] = min_distance
            lb[gi, i] = group_min_distance2
            group_id = gi
            min_distance = group_min_distance
            label = group_label
        else
            lb[gi, i] = group_min_distance
        end
    end

    ub[i] = min_distance
    labels[i] = label

    return label
end

@inline function update_bounds(alg::YingYang, ub, lb, labels, p, groups, gd, i)
    # Since bounds are squred distance, `sqrt` is used to make corresponding estimation, unlike
    # the original paper, where usual metric is used.
    #
    # If `u` is upper bound and `a` is `labels` then
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

    # TODO: another version of this function for all other metrics
    ub[i] += 2*sqrt(abs(ub[i] * p[labels[i]])) + p[labels[i]]
    for gi in 1:length(groups)
        lb[gi, i] += gd[gi] - 2*sqrt(abs(gd[gi] * lb[gi, i]))
    end
end


## Misc
# Borrowed from https://github.com/JuliaLang/julia/pull/21598/files

# swap columns i and j of a, in-place
function swapcols!(a, i, j)
    i == j && return
    for k in axes(a,1)
        @inbounds a[k,i], a[k,j] = a[k,j], a[k,i]
    end
end

# like permute!! applied to each row of a, in-place in a (overwriting p).
function permutecols!!(a, p)
    count = 0
    start = 0
    while count < length(p)
        ptr = start = findnext(!iszero, p, start+1)::Int
        next = p[start]
        count += 1
        while next != start
            swapcols!(a, ptr, next)
            p[ptr] = 0
            ptr = next
            next = p[next]
            count += 1
        end
        p[ptr] = 0
    end
    a
end
