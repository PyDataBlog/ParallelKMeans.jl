# All Abstract types defined
"""
    AbstractKMeansAlg

Abstract base type inherited by all sub-KMeans algorithms.
"""
abstract type AbstractKMeansAlg end


"""
    ClusteringResult

Base type for the output of clustering algorithm.
"""
abstract type ClusteringResult end


# Here we mimic `Clustering` output structure
"""
    KmeansResult{C,D<:Real,WC<:Real} <: ClusteringResult

The output of [`kmeans`](@ref) and [`kmeans!`](@ref).
# Type parameters
 * `C<:AbstractMatrix{<:AbstractFloat}`: type of the `centers` matrix
 * `D<:Real`: type of the assignment cost
 * `WC<:Real`: type of the cluster weight
 # C is the type of centers, an (abstract) matrix of size (d x k)
# D is the type of pairwise distance computation from points to cluster centers
# WC is the type of cluster weights, either Int (in the case where points are
# unweighted) or eltype(weights) (in the case where points are weighted).
"""
struct KmeansResult{C<:AbstractMatrix{<:AbstractFloat},D<:Real,WC<:Real} <: ClusteringResult
    centers::C                 # cluster centers (d x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{D}           # cost of the assignments (n)
    counts::Vector{Int}        # number of points assigned to each cluster (k)
    wcounts::Vector{WC}        # cluster weights (k)
    totalcost::D               # total cost (i.e. objective)
    iterations::Int            # number of elapsed iterations
    converged::Bool            # whether the procedure converged
end

"""
    sum_of_squares(x, labels, centre, k)

This function computes the total sum of squares based on the assigned (labels)
design matrix(x), centroids (centre), and the number of desired groups (k).

A Float type representing the computed metric is returned.
"""
function sum_of_squares(x, labels, centre)
    s = 0.0

    @inbounds for j in axes(x, 2)
        for i in axes(x, 1)
            s += (x[i, j] - centre[i, labels[j]])^2
        end
    end

    return s
end


"""
    Kmeans([alg::AbstractKMeansAlg,] design_matrix, k; n_threads = nthreads(), k_init="k-means++", max_iters=300, tol=1e-6, verbose=true)

This main function employs the K-means algorithm to cluster all examples
in the training data (design_matrix) into k groups using either the
`k-means++` or random initialisation technique for selecting the initial
centroids.

At the end of the number of iterations specified (max_iters), convergence is
achieved if difference between the current and last cost objective is
less than the tolerance level (tol). An error is thrown if convergence fails.

Arguments:
- `alg` defines one of the algorithms used to calculate `k-means`. This
argument can be omitted, by default Lloyd algorithm is used.
- `n_threads` defines number of threads used for calculations, by default it is equal
to the `Threads.nthreads()` which is defined by `JULIA_NUM_THREADS` environmental
variable. For small size design matrices it make sense to set this argument to 1 in order
to avoid overhead of threads generation.
- `k_init` is one of the algorithms used for initialization. By default `k-means++` algorithm is used,
alternatively one can use `rand` to choose random points for init.
- `max_iters` is the maximum number of iterations
- `tol` defines tolerance for early stopping.
- `verbose` is verbosity level. Details of operations can be either printed or not by setting verbose accordingly.

A `KmeansResult` structure representing labels, centroids, and sum_squares is returned.
"""
function kmeans(alg, design_matrix, k;
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = 1e-6, verbose = false, init = nothing)
    nrow, ncol = size(design_matrix)
    containers = create_containers(alg, k, nrow, ncol, n_threads)

    return kmeans!(alg, containers, design_matrix, k, n_threads = n_threads,
                    k_init = k_init, max_iters = max_iters, tol = tol,
                    verbose = verbose, init = init)
end


"""
    Kmeans!(alg::AbstractKMeansAlg, containers, design_matrix, k; n_threads = nthreads(), k_init="k-means++", max_iters=300, tol=1e-6, verbose=true)

Mutable version of `kmeans` function. Definition of arguments and results can be
found in `kmeans`.

Argument `containers` represent algorithm specific containers, such as labels, intermidiate
centroids and so on, which are used during calculations.
"""
function kmeans!(alg, containers, design_matrix, k;
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = 1e-6, verbose = false, init = nothing)
    nrow, ncol = size(design_matrix)
    centroids = init == nothing ? smart_init(design_matrix, k, n_threads, init=k_init).centroids : deepcopy(init)

    converged = false
    niters = 1
    J_previous = 0.0

    # Update centroids & labels with closest members until convergence

    while niters <= max_iters
        update_containers!(containers, alg, centroids, n_threads)
        J = update_centroids!(centroids, containers, alg, design_matrix, n_threads)

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

    totalcost = sum_of_squares(design_matrix, containers.labels, centroids)

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
    update_centroids!(centroids, containers, alg, design_matrix, n_threads)

Internal function, used to update centroids by utilizing one of `alg`. It works as
a wrapper of internal `chunk_update_centroids!` function, splitting incoming
`design_matrix` in chunks and combining results together.
"""
function update_centroids!(centroids, containers, alg, design_matrix, n_threads)
    ncol = size(design_matrix, 2)

    if n_threads == 1
        r = axes(design_matrix, 2)
        J = chunk_update_centroids!(centroids, containers, alg, design_matrix, r, 1)

        centroids .= containers.new_centroids[1] ./ containers.centroids_cnt[1]'
    else
        ranges = splitter(ncol, n_threads)

        waiting_list = Vector{Task}(undef, n_threads - 1)

        for i in 1:length(ranges) - 1
            waiting_list[i] = @spawn chunk_update_centroids!(centroids, containers,
                alg, design_matrix, ranges[i], i + 1)
        end

        J = chunk_update_centroids!(centroids, containers, alg, design_matrix, ranges[end], 1)

        J += sum(fetch.(waiting_list))

        for i in 1:length(ranges) - 1
            containers.new_centroids[1] .+= containers.new_centroids[i + 1]
            containers.centroids_cnt[1] .+= containers.centroids_cnt[i + 1]
        end

        centroids .= containers.new_centroids[1] ./ containers.centroids_cnt[1]'
    end

    return J/ncol
end
