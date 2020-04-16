"""
    Elkan()

Elkan algorithm implementation, based on "Charles Elkan. 2003.
Using the triangle inequality to accelerate k-means.
In Proceedings of the Twentieth International Conference on
International Conference on Machine Learning (ICML’03). AAAI Press, 147–153."

This algorithm provides much faster convergence than Lloyd algorithm especially
for high dimensional data.
It can be used directly in `kmeans` function

```julia
X = rand(30, 100_000)   # 100_000 random points in 30 dimensions

kmeans(Elkan(), X, 3) # 3 clusters, Elkan algorithm
```
"""
struct Elkan <: AbstractKMeansAlg end

function kmeans!(alg::Elkan, containers, X, k;
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = 1e-6, verbose = false, init = nothing)
    nrow, ncol = size(X)
    centroids = init == nothing ? smart_init(X, k, n_threads, init=k_init).centroids : deepcopy(init)

    update_containers(alg, containers, centroids, n_threads)
    @parallelize n_threads ncol chunk_initialize(alg, containers, centroids, X)

    converged = false
    niters = 0
    J_previous = 0.0

    # Update centroids & labels with closest members until convergence
    while niters < max_iters
        niters += 1
        # Core iteration
        @parallelize n_threads ncol chunk_update_centroids(alg, containers, centroids, X)

        # Collect distributed containers (such as centroids_new, centroids_cnt)
        # in paper it is step 4
        collect_containers(alg, containers, n_threads)

        J = sum(containers.ub)

        # auxiliary calculation, in paper it's d(c, m(c))
        calculate_centroids_movement(alg, containers, centroids)

        # lower and ounds update, in paper it's steps 5 and 6
        @parallelize n_threads ncol chunk_update_bounds(alg, containers, centroids)

        # Step 7, final assignment of new centroids
        centroids .= containers.centroids_new[end]

        if verbose
            # Show progress and terminate if J stopped decreasing.
            println("Iteration $niters: Jclust = $J")
        end

        # Check for convergence
        if (niters > 1) & (abs(J - J_previous) < (tol * J))
            converged = true
            break
        end

        # Step 1 in original paper, calulation of distance d(c, c')
        update_containers(alg, containers, centroids, n_threads)
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

function create_containers(::Elkan, k, nrow, ncol, n_threads)
    lng = n_threads + 1
    centroids_new = Vector{Array{Float64,2}}(undef, lng)
    centroids_cnt = Vector{Vector{Int}}(undef, lng)

    for i = 1:lng
        centroids_new[i] = zeros(nrow, k)
        centroids_cnt[i] = zeros(k)
    end

    centroids_dist = Matrix{Float64}(undef, k, k)

    # lower bounds
    lb = Matrix{Float64}(undef, k, ncol)

    # upper bounds
    ub = Vector{Float64}(undef, ncol)

    # r(x) in original paper, shows whether point distance should be updated
    stale = ones(Bool, ncol)

    # distance that centroid moved
    p = Vector{Float64}(undef, k)

    labels = zeros(Int, ncol)

    # total_sum_calculation
    sum_of_squares = Vector{Float64}(undef, n_threads)

    return (
        centroids_new = centroids_new,
        centroids_cnt = centroids_cnt,
        labels = labels,
        centroids_dist = centroids_dist,
        lb = lb,
        ub = ub,
        stale = stale,
        p = p,
        sum_of_squares = sum_of_squares
    )
end

function chunk_initialize(::Elkan, containers, centroids, X, r, idx)
    ub = containers.ub
    lb = containers.lb
    centroids_dist = containers.centroids_dist
    labels = containers.labels
    centroids_new = containers.centroids_new[idx]
    centroids_cnt = containers.centroids_cnt[idx]

    @inbounds for i in r
        min_dist = distance(X, centroids, i, 1)
        label = 1
        lb[label, i] = min_dist
        for j in 2:size(centroids, 2)
            # triangular inequality
            if centroids_dist[j, label] > min_dist
                lb[j, i] = min_dist
            else
                dist = distance(X, centroids, i, j)
                label = dist < min_dist ? j : label
                min_dist = dist < min_dist ? dist : min_dist
                lb[j, i] = dist
            end
        end
        ub[i] = min_dist
        labels[i] = label
        centroids_cnt[label] += 1
        for j in axes(X, 1)
            centroids_new[j, label] += X[j, i]
        end
    end
end

function update_containers(::Elkan, containers, centroids, n_threads)
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

function chunk_update_centroids(::Elkan, containers, centroids, X, r, idx)
    # unpack
    ub = containers.ub
    lb = containers.lb
    centroids_dist = containers.centroids_dist
    labels = containers.labels
    stale = containers.stale
    centroids_new = containers.centroids_new[idx]
    centroids_cnt = containers.centroids_cnt[idx]

    @inbounds for i in r
        label_old = labels[i]
        label = label_old
        min_dist = ub[i]
        # tighten the loop, exclude points that very close to center
        min_dist <= centroids_dist[label, label] && continue
        for j in axes(centroids, 2)
            # tighten the loop once more, exclude far away centers
            j == label && continue
            min_dist <= lb[j, i] && continue
            min_dist <= centroids_dist[j, label] && continue

            # one calculation per iteration is enough
            if stale[i]
                min_dist = distance(X, centroids, i, label)
                lb[label, i] = min_dist
                ub[i] = min_dist
                stale[i] = false
            end

            if (min_dist > lb[j, i]) | (min_dist > centroids_dist[j, label])
                dist = distance(X, centroids, i, j)
                lb[j, i] = dist
                if dist < min_dist
                    min_dist = dist
                    label = j
                end
            end
        end

        if label != label_old
            labels[i] = label
            centroids_cnt[label_old] -= 1
            centroids_cnt[label] += 1
            for j in axes(X, 1)
                centroids_new[j, label_old] -= X[j, i]
                centroids_new[j, label] += X[j, i]
            end
        end
    end
end

function calculate_centroids_movement(alg::Elkan, containers, centroids)
    p = containers.p
    centroids_new = containers.centroids_new[end]

    for i in axes(centroids, 2)
        p[i] = distance(centroids, centroids_new, i, i)
    end
end

function chunk_update_bounds(alg, containers, centroids, r, idx)
    p = containers.p
    lb = containers.lb
    ub = containers.ub
    stale = containers.stale
    labels = containers.labels

    @inbounds for i in r
        for j in axes(centroids, 2)
            lb[j, i] = lb[j, i] > p[j] ? lb[j, i] + p[j] - 2*sqrt(abs(lb[j, i]*p[j])) : 0.0
        end
        stale[i] = true
        ub[i] += p[labels[i]] + 2*sqrt(abs(ub[i]*p[labels[i]]))
    end
end
