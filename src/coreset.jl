"""
    Coreset()

Coreset algorithm implementation, based on "Lucic, Mario & Bachem,
Olivier & Krause, Andreas. (2015). Strong Coresets for Hard and Soft Bregman
Clustering with Applications to Exponential Family Mixtures."

`Coreset` supports following arguments:
- `m`: default 100, subsample size
- `alg`: default `Lloyd()`, algorithm used to clusterize sample

It can be used directly in `kmeans` function

```julia
X = rand(30, 100_000)   # 100_000 random points in 30 dimensions

# 3 clusters, Coreset algorithm with default Lloyd algorithm and 100 subsamples
kmeans(Coreset(), X, 3)

# 3 clusters, Coreset algorithm with Hamerly algorithm and 500 subsamples
kmeans(Coreset(m = 500, alg = Hamerly()), X, 3)
kmeans(Coreset(500, Hamerly()), X, 3)

# alternatively short form can be used for defining subsample size or algorithm only
kmeans(Coreset(500), X, 3) # sample of the size 500, Lloyd clustering algorithm
kmeans(Coreset(Hamerly()), X, 3) # sample of the size 100, Hamerly clustering algorithm
```
"""
struct Coreset{T <: AbstractKMeansAlg} <: AbstractKMeansAlg
    m::Int
    alg::T
end

Coreset(; m = 100, alg = Lloyd()) = Coreset(m, alg)
Coreset(m::Int) = Coreset(m, Lloyd())
Coreset(alg::AbstractKMeansAlg) = Coreset(100, alg)

function kmeans!(alg::Coreset, containers, X, k, weights;
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = eltype(design_matrix)(1e-6), verbose = false,
                init = nothing, rng = Random.GLOBAL_RNG)
    nrow, ncol = size(X)
    centroids = isnothing(init) ? smart_init(X, k, n_threads, weights, rng, init=k_init).centroids : deepcopy(init)

    T = eltype(X)
    # Steps 2-4 of the paper's algorithm 3
    # We distribute points over the centers and calculate weights of each cluster
    @parallelize n_threads ncol chunk_fit(alg, containers, centroids, X, weights)

    # after this step, containers.centroids_new
    collect_containers(alg, containers, n_threads)

    # step 7 of the algorithm 3
    @parallelize n_threads ncol chunk_update_sensitivity(alg, containers)

    # sample from containers.s
    coreset_ids = wsample(rng, 1:ncol, containers.s, alg.m)
    coreset = X[:, coreset_ids]
    # create new weights as 1/s[i]
    coreset_weights = one(T) ./ @view containers.s[coreset_ids]

    # run usual kmeans for new set with new weights.
    res = kmeans(alg.alg, coreset, k, weights = coreset_weights, tol = tol, max_iters = max_iters,
        verbose = verbose, init = centroids, n_threads = n_threads, rng = rng)

    @parallelize n_threads ncol chunk_apply(alg, containers, res.centers, X, weights)

    totalcost = sum(containers.totalcost)

    return KmeansResult(res.centers, containers.labels, T[], Int[], T[], totalcost, res.iterations, res.converged)
end

function create_containers(alg::Coreset, X, k, nrow, ncol, n_threads)
    T = eltype(X)

    centroids_cnt = Vector{Vector{T}}(undef, n_threads)
    centroids_dist = Vector{Vector{T}}(undef, n_threads)

    # sensitivity

    for i in 1:n_threads
        centroids_cnt[i] = zeros(T, k)
        centroids_dist[i] = zeros(T, k)
    end

    labels = Vector{Int}(undef, ncol)
    s = Vector{T}(undef, ncol)

    # J is the same as $c_\phi$ in the paper.
    J = Vector{T}(undef, n_threads)

    alpha = 16 * (log(k) + 2)

    centroids_const = Vector{T}(undef, k)

    # total_sum_calculation
    totalcost = Vector{T}(undef, n_threads)

    return (
        centroids_cnt = centroids_cnt,
        centroids_dist = centroids_dist,
        s = s,
        labels = labels,
        totalcost = totalcost,
        J = J,
        centroids_const = centroids_const,
        alpha = alpha
    )
end

function chunk_fit(alg::Coreset, containers, centroids, X, weights, r, idx)
    centroids_cnt = containers.centroids_cnt[idx]
    centroids_dist = containers.centroids_dist[idx]
    labels = containers.labels
    s = containers.s
    T = eltype(X)

    J = zero(T)
    for i in r
        dist = distance(X, centroids, i, 1)
        label = 1
        for j in 2:size(centroids, 2)
            new_dist = distance(X, centroids, i, j)

            # calculation of the closest center (steps 2-3 of the paper's algorithm 3)
            label = new_dist < dist ? j : label
            dist = new_dist < dist ? new_dist : dist
        end
        labels[i] = label

        # calculation of the $c_\phi$ (step 4)
        # Note: $d_A(x', B) = min_{b \in B} d_A(x', b)$
        # Not exactly sure about whole `weights` thing, needs further investigation
        # for Nothing `weights` (default) it'll work as intendent
        centroids_cnt[label] += isnothing(weights) ? one(T) : weights[i]
        centroids_dist[label] += isnothing(weights) ? dist : weights[i] * dist
        J += dist

        # for now we write dist to sensitivity, update it later
        s[i] = dist
    end

    containers.J[idx] = J
end

function collect_containers(::Coreset, containers, n_threads)
    # Here we transform formula of the step 6
    # By multiplying both sides of equation on $c_\phi / \alpha$ we obtain
    # $s(x) <- d_A(x, B) + 2 \sum d_A(x, B) / |B_i| + 4 c_\phi |\Chi| / (|B_i| * \alpha)$
    # Taking into account that $c_\phi = 1/|\Chi| \sum d_A(x', B) = J / |\Chi|$ we get
    # $s(x) <- d_A(x, B) + 2 \sum d_A(x, B) / |B_i| + 4 J / \alpha * 1/ |B_i|$

    alpha = containers.alpha
    centroids_const = containers.centroids_const

    centroids_cnt = containers.centroids_cnt[1]
    centroids_dist = containers.centroids_dist[1]
    J = containers.J[1]

    @inbounds for i in 2:n_threads
        centroids_cnt .+= containers.centroids_cnt[i]
        centroids_dist .+= containers.centroids_dist[i]
        J += containers.J[i]
    end

    J = 4 * J / alpha

    for i in 1:length(centroids_dist)
        centroids_const[i] = 2 * centroids_dist[i] / centroids_cnt[i] +
            J / centroids_cnt[i]
    end
end

function chunk_update_sensitivity(alg::Coreset, containers, r, idx)
    labels = containers.labels
    centroids_const = containers.centroids_const
    s = containers.s

    @inbounds for i in r
        s[i] += centroids_const[labels[i]]
    end
end

function chunk_apply(alg::Coreset, containers, centroids, X, weights, r, idx)
    centroids_cnt = containers.centroids_cnt[idx]
    centroids_dist = containers.centroids_dist[idx]
    labels = containers.labels
    T = eltype(X)

    J = zero(T)
    for i in r
        dist = distance(X, centroids, i, 1)
        label = 1
        for j in 2:size(centroids, 2)
            new_dist = distance(X, centroids, i, j)

            # calculation of the closest center (steps 2-3 of the paper's algorithm 3)
            label = new_dist < dist ? j : label
            dist = new_dist < dist ? new_dist : dist
        end
        labels[i] = label
        J += isnothing(weights) ? dist : weights[i] * dist
    end

    containers.totalcost[idx] = J
end
