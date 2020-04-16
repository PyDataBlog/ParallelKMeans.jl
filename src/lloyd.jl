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
function kmeans!(alg::Lloyd, containers, X, k;
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

kmeans(design_matrix, k;
    n_threads = Threads.nthreads(),
    k_init = "k-means++", max_iters = 300, tol = 1e-6,
    verbose = false, init = nothing) =
        kmeans(Lloyd(), design_matrix, k; n_threads = n_threads, k_init = k_init, max_iters = max_iters, tol = tol,
            verbose = verbose, init = init)

"""
    create_containers(::Lloyd, k, nrow, ncol, n_threads)

Internal function for the creation of all necessary intermidiate structures.

- `centroids_new` - container which holds new positions of centroids
- `centroids_cnt` - container which holds number of points for each centroid
- `labels` - vector which holds labels of corresponding points
"""
function create_containers(::Lloyd, k, nrow, ncol, n_threads)
    lng = n_threads + 1
    centroids_new = Vector{Array{Float64,2}}(undef, lng)
    centroids_cnt = Vector{Vector{Int}}(undef, lng)

    for i in 1:lng
        centroids_new[i] = Array{Float64, 2}(undef, nrow, k)
        centroids_cnt[i] = Vector{Int}(undef, k)
    end

    labels = Vector{Int}(undef, ncol)

    J = Vector{Float64}(undef, n_threads)

    # total_sum_calculation
    sum_of_squares = Vector{Float64}(undef, n_threads)

    return (centroids_new = centroids_new, centroids_cnt = centroids_cnt,
            labels = labels, J = J, sum_of_squares = sum_of_squares)
end

function chunk_update_centroids(::Lloyd, containers, centroids, X, r, idx)
    # unpack containers for easier manipulations
    centroids_new = containers.centroids_new[idx]
    centroids_cnt = containers.centroids_cnt[idx]
    labels = containers.labels

    centroids_new .= 0.0
    centroids_cnt .= 0
    J = 0.0
    @inbounds for i in r
        min_dist = distance(X, centroids, i, 1)
        label = 1
        for j in 2:size(centroids, 2)
            dist = distance(X, centroids, i, j)
            label = dist < min_dist ? j : label
            min_dist = dist < min_dist ? dist : min_dist
        end
        labels[i] = label
        centroids_cnt[label] += 1
        for j in axes(X, 1)
            centroids_new[j, label] += X[j, i]
        end
        J += min_dist
    end

    containers.J[idx] = J
end
