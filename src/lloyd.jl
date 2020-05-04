"""
    Lloyd <: AbstractKMeansAlg

Basic algorithm for k-means calculation.
"""
struct Lloyd <: AbstractKMeansAlg end

"""
    Kmeans!(alg::AbstractKMeansAlg, containers, design_matrix, k; n_threads = nthreads(), k_init="k-means++", max_iters=300, tol=1e-6, verbose=true)

Mutable version of `kmeans` function. Definition of arguments and results can be
found in `kmeans`.

Argument `containers` represent algorithm specific containers, such as labels, intermidiate
centroids and so on, which are used during calculations.
"""
function kmeans!(alg::Lloyd, containers, X, k, weights;
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = eltype(design_matrix)(1e-6), verbose = false,
                init = nothing, rng = Random.GLOBAL_RNG, metric=Euclidean())

    nrow, ncol = size(X)
    centroids = isnothing(init) ? smart_init(X, k, n_threads, weights, rng, init=k_init).centroids : deepcopy(init)

    T = eltype(X)
    converged = false
    niters = 1
    J_previous = zero(T)

    # Update centroids & labels with closest members until convergence
    while niters <= max_iters
        @parallelize n_threads ncol chunk_update_centroids(alg, containers, centroids, X, weights, metric)
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
        niters += 1  # TODO: Investigate the potential bug in number of iterations
    end
    @parallelize n_threads ncol sum_of_squares(containers, X, containers.labels, centroids, weights, metric)
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

kmeans(design_matrix, k;
    weights = nothing,
    n_threads = Threads.nthreads(),
    k_init = "k-means++", max_iters = 300, tol = 1e-6,
    verbose = false, init = nothing, rng = Random.GLOBAL_RNG, metric = Euclidean()) =
        kmeans(Lloyd(), design_matrix, k; weights = weights, n_threads = n_threads, k_init = k_init, max_iters = max_iters, tol = tol,
            verbose = verbose, init = init, rng = rng, metric = metric)


"""
    create_containers(::Lloyd, k, nrow, ncol, n_threads)

Internal function for the creation of all necessary intermidiate structures.

- `centroids_new` - container which holds new positions of centroids
- `centroids_cnt` - container which holds number of points for each centroid
- `labels` - vector which holds labels of corresponding points
"""
function create_containers(::Lloyd, X, k, nrow, ncol, n_threads)
    T = eltype(X)
    lng = n_threads + 1
    centroids_new = Vector{Matrix{T}}(undef, lng)
    centroids_cnt = Vector{Vector{T}}(undef, lng)

    for i in 1:lng
        centroids_new[i] = Matrix{T}(undef, nrow, k)
        centroids_cnt[i] = Vector{Int}(undef, k)
    end

    labels = Vector{Int}(undef, ncol)

    J = Vector{T}(undef, n_threads)

    # total_sum_calculation
    sum_of_squares = Vector{T}(undef, n_threads)

    return (centroids_new = centroids_new, centroids_cnt = centroids_cnt,
            labels = labels, J = J, sum_of_squares = sum_of_squares)
end


function chunk_update_centroids(::Lloyd, containers, centroids, X, weights, metric, r, idx)
    # unpack containers for easier manipulations
    centroids_new = containers.centroids_new[idx]
    centroids_cnt = containers.centroids_cnt[idx]
    labels = containers.labels
    T = eltype(X)

    centroids_new .= zero(T)
    centroids_cnt .= zero(T)
    J = zero(T)
    @inbounds for i in r
        min_dist = distance(metric, X, centroids, i, 1)
        label = 1
        for j in 2:size(centroids, 2)
            dist = distance(metric, X, centroids, i, j)
            label = dist < min_dist ? j : label
            min_dist = dist < min_dist ? dist : min_dist
        end
        labels[i] = label
        centroids_cnt[label] += isnothing(weights) ? one(T) : weights[i]
        for j in axes(X, 1)
            centroids_new[j, label] += isnothing(weights) ? X[j, i] : weights[i] * X[j, i]
        end
        J += min_dist
    end

    containers.J[idx] = J
end


function collect_containers(alg::Lloyd, containers, centroids, n_threads)
    if n_threads == 1
        @inbounds centroids .= containers.centroids_new[1] ./ containers.centroids_cnt[1]'
    else
        @inbounds containers.centroids_new[end] .= containers.centroids_new[1]
        @inbounds containers.centroids_cnt[end] .= containers.centroids_cnt[1]
        @inbounds for i in 2:n_threads
            containers.centroids_new[end] .+= containers.centroids_new[i]
            containers.centroids_cnt[end] .+= containers.centroids_cnt[i]
        end

        @inbounds centroids .= containers.centroids_new[end] ./ containers.centroids_cnt[end]'
    end
end
