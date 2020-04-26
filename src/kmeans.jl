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
    spliiter(n, k)

Internal utility function, splits 1:n sequence to k chunks of approximately same size.
"""
function splitter(n, k)
    xz = Int.(ceil.(range(0, n, length = k+1)))
    return [xz[i]+1:xz[i+1] for i in 1:k]
end

"""
    @parallelize(n_threads, ncol, f)

Parallelize function and run it over n_threads. Function should require following conditions:
1. It should not return any values.
1. It should accept parameters two parameters at the end of the argument list. First
accepted parameter is `range`, which defines chunk used in calculations. Second
parameter is `idx` which defines id of the container where results can be stored.

`ncol` argument defines range 1:ncol which is sliced in `n_threads` chunks.
"""
macro parallelize(n_threads, ncol, f)
    for i in 1:length(f.args)
        f.args[i] = :($(esc(f.args[i])))
    end
    single_thread_chunk = copy(f)
    push!(single_thread_chunk.args, :(1:$(esc(ncol))))
    push!(single_thread_chunk.args, 1)

    multi_thread_chunk = copy(f)
    push!(multi_thread_chunk.args, :(ranges[i]))
    push!(multi_thread_chunk.args, :(i))

    last_multi_thread_chunk = copy(f)
    push!(last_multi_thread_chunk.args, :(ranges[end]))
    push!(last_multi_thread_chunk.args, :($(esc(n_threads))))

    return quote
        if $(esc(n_threads)) == 1
            $single_thread_chunk
        else
            local ranges = splitter($(esc(ncol)), $(esc(n_threads)))
            local waiting_list = $(esc(Vector)){$(esc(Task))}(undef, $(esc(n_threads)) - 1)
            for i in 1:$(esc(n_threads)) - 1
                waiting_list[i] = @spawn $multi_thread_chunk
            end

            $last_multi_thread_chunk

            for i in 1:$(esc(n_threads)) - 1
                wait(waiting_list[i])
            end
        end
    end
end


"""
    distance(X1, X2, i1, i2)

Allocationless calculation of square eucledean distance between vectors X1[:, i1] and X2[:, i2]
"""
function distance(X1, X2, i1, i2)
    d = zero(eltype(X1))
    # TODO: break of the loop if d is larger than threshold (known minimum disatnce)
    @inbounds @simd for i in axes(X1, 1)
        d += (X1[i, i1] - X2[i, i2])^2
    end
    return d
end


"""
    distance(metric, X1, X2, i1, i2)

Allocationless calculation of distance between vectors X1[:, i1] and X2[:, i2] defined by the supplied distance metric.
"""
distance(metric, X1, X2, i1, i2) = evaluate(metric, X1[:, i1], X2[:, i2])


"""
    distance(X1, X2, i1, i2)

Allocationless calculation of square eucledean distance between vectors X1[:, i1] and X2[:, i2]
"""
function distance(metric::Euclidean, X1, X2, i1, i2)
    # here goes my definition
    d = zero(eltype(X1))
    # TODO: break of the loop if d is larger than threshold (known minimum disatnce)
    @inbounds @simd for i in axes(X1, 1)
        d += (X1[i, i1] - X2[i, i2])^2
    end

    return d
end


"""
    sum_of_squares(x, labels, centre, k)

This function computes the total sum of squares based on the assigned (labels)
design matrix(x), centroids (centre), and the number of desired groups (k).

A Float type representing the computed metric is returned.
"""
function sum_of_squares(containers, x, labels, centre, weights, r, idx)
    s = zero(eltype(x))

    @inbounds for i in r
        s += isnothing(weights) ? distance(x, centre, i, labels[i]) : weights[i] * distance(x, centre, i, labels[i])
    end

    containers.sum_of_squares[idx] = s
end


"""
    kmeans([alg::AbstractKMeansAlg,] design_matrix, k; n_threads = nthreads(),
    k_init="k-means++", max_iters=300, tol=1e-6, verbose=true, rng = Random.GLOBAL_RNG)

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
function kmeans(alg::AbstractKMeansAlg, design_matrix, k;
                weights = nothing,
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = eltype(design_matrix)(1e-6), verbose = false,
                init = nothing, rng = Random.GLOBAL_RNG)
    nrow, ncol = size(design_matrix)
    containers = create_containers(alg, design_matrix, k, nrow, ncol, n_threads)

    return kmeans!(alg, containers, design_matrix, k, weights, n_threads = n_threads,
                    k_init = k_init, max_iters = max_iters, tol = tol,
                    verbose = verbose, init = init, rng = rng)
end


function collect_containers(::AbstractKMeansAlg, containers, n_threads)
    if n_threads == 1
        @inbounds containers.centroids_new[end] .= containers.centroids_new[1] ./ containers.centroids_cnt[1]'
    else
        @inbounds containers.centroids_new[end] .= containers.centroids_new[1]
        @inbounds containers.centroids_cnt[end] .= containers.centroids_cnt[1]
        @inbounds for i in 2:n_threads
            containers.centroids_new[end] .+= containers.centroids_new[i]
            containers.centroids_cnt[end] .+= containers.centroids_cnt[i]
        end

        @inbounds containers.centroids_new[end] .= containers.centroids_new[end] ./ containers.centroids_cnt[end]'
    end
end
