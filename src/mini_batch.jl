"""
    MiniBatch(b::Int)
    `b` represents the size of the batch which should be sampled.

    Sculley et al. 2007 Mini batch k-means algorithm implementation.

```julia
X = rand(30, 100_000)  # 100_000 random points in 30 dimensions

kmeans(MiniBatch(100), X, 3)  # 3 clusters, MiniBatch algorithm with 100 batch samples at each iteration
```
"""
mutable struct MiniBatch <: AbstractKMeansAlg
    b::Int  # batch size
end


MiniBatch() = MiniBatch(100)

function kmeans!(alg::MiniBatch, containers, X, k,
                 weights = nothing, metric = Euclidean(); n_threads = Threads.nthreads(),
                 k_init = "k-means++", init = nothing, max_iters = 300,
                 tol = eltype(X)(1e-6), max_no_improvement = 10, verbose = false, rng = Random.GLOBAL_RNG)

    # Retrieve initialized artifacts from the container
    centroids = containers.centroids_new
    batch_rand_idx = containers.batch_rand_idx
    labels = containers.labels

    # Get the type and dimensions of design matrix, X - (Step 1)
    T = eltype(X)
    nrow, ncol = size(X)

    # Initiate cluster centers - (Step 2) in paper
    centroids .= isnothing(init) ? smart_init(X, k, n_threads, weights, rng, init = k_init).centroids : deepcopy(init)

    # Initialize counter for the no. of data in each cluster - (Step 3) in paper
    N = zeros(T, k)

    # Initialize various artifacts
    converged = false
    niters = 1
    counter = 0
    J_previous = zero(T)
    J = zero(T)
    totalcost = zero(T)
    prev_labels = copy(labels)
    prev_centroids = copy(centroids)

    # Main Steps. Batch update centroids until convergence
    while niters <= max_iters  # Step 4 in paper

        # b examples picked randomly from X (Step 5 in paper)
        isnothing(weights) ? rand!(rng, batch_rand_idx, 1:ncol) : wsample!(rng, 1:ncol, weights, batch_rand_idx)

        # Cache/label the batch samples nearest to the centers (Step 6 & 7)
        @inbounds for i in batch_rand_idx
            min_dist = distance(metric, X, centroids, i, 1)
            label = 1

            for j in 2:size(centroids, 2)
                dist = distance(metric, X, centroids, i, j)
                label = dist < min_dist ? j : label
                min_dist = dist < min_dist ? dist : min_dist
            end

            labels[i] = label

            ##### Batch gradient step  #####
            # iterate over examples (each column) ==> (Step 9)
            # Get cached center/label for each example label = labels[i] => (Step 10)

            # Update per-center counts
            N[label] += isnothing(weights) ? 1 : weights[i]  # (Step 11)

            # Get per-center learning rate (Step 12)
            lr = 1 / N[label]

            # Take gradient step (Step 13) # TODO: Replace with faster loop?
            @views centroids[:, label] .= (1 - lr) .* centroids[:, label] .+ (lr .* X[:, i])
        end

        # Reassign all labels based on new centres generated from the latest sample
        labels .= reassign_labels(X, metric, labels, centroids)

        # Calculate cost on whole dataset after reassignment and check for convergence
        @parallelize 1 ncol sum_of_squares(containers, X, labels, centroids, weights, metric)
        J = sum(containers.sum_of_squares)

        if verbose
            # Show progress and terminate if J stopped decreasing.
            println("Iteration $niters: Jclust = $J")
        end

        # Check for early stopping convergence
        if (niters > 1) & (abs(J - J_previous) < (tol * J))
            counter += 1

            # Declare convergence if max_no_improvement criterion is met
            if counter >= max_no_improvement
                converged = true
                # Compute label assignment for the complete dataset
                labels .= reassign_labels(X, metric, labels, centroids)

                # Compute totalcost for the complete dataset
                @parallelize 1 ncol sum_of_squares(containers, X, labels, centroids, weights, metric)
                totalcost = sum(containers.sum_of_squares)

                # Print convergence message to user
                if verbose
                    println("Successfully terminated with convergence.")
                end

                break
            end
        else
            counter = 0
        end

        # Adaptive batch size mechanism
        if counter > 0
            alg.b = min(alg.b * 2, ncol)
        else
            alg.b = max(alg.b ÷ 2, 1)
        end

        # Early stopping criteria based on change in cluster assignments
        if labels == prev_labels && all(centroids .== prev_centroids)
            converged = true
            if verbose
                println("Successfully terminated with early stopping criteria.")
            end
            break
        end

        prev_labels .= labels
        prev_centroids .= centroids

        # Warn users if model doesn't converge at max iterations
        if (niters >= max_iters) & (!converged)

            if verbose
                println("Clustering model failed to converge. Labelling data with latest centroids.")
            end

            labels .= reassign_labels(X, metric, labels, centroids)

            # Compute totalcost for unconverged model
            @parallelize 1 ncol sum_of_squares(containers, X, labels, centroids, weights, metric)
            totalcost = sum(containers.sum_of_squares)

            break
        end

        J_previous = J
        niters += 1
    end

    # Push learned artifacts to KmeansResult
    return KmeansResult(centroids, labels, T[], Int[], T[], totalcost, niters, converged)
end

"""
    reassign_labels(DMatrix, metric, labels, centres)

An internal function to relabel DMatrix based on centres and metric.
"""
function reassign_labels(DMatrix, metric, labels, centres)
    @inbounds for i in axes(DMatrix, 2)
        min_dist = distance(metric, DMatrix, centres, i, 1)
        label = 1

        for j in 2:size(centres, 2)
            dist = distance(metric, DMatrix, i, j)
            label = dist < min_dist ? j : label
            min_dist = dist < min_dist ? dist : min_dist
        end

        labels[i] = label
    end
    return labels
end

"""
    create_containers(::MiniBatch, k, nrow, ncol, n_threads)

Internal function for the creation of all necessary intermidiate structures.

- `centroids_new` - container which holds new positions of centroids
- `labels` - vector which holds labels of corresponding points
- `sum_of_squares` - vector which holds the sum of squares values for each thread
- `batch_rand_idx` - vector which holds the selected batch indices
"""
function create_containers(alg::MiniBatch, X, k, nrow, ncol, n_threads)
    # Initiate placeholders to avoid allocations
    T = eltype(X)
    labels = Vector{Int}(undef, ncol)  # labels vector
    sum_of_squares = Vector{T}(undef, 1)  # total_sum_calculation
    batch_rand_idx = Vector{Int}(undef, alg.b)  # selected batch indices
    centroids_new = Matrix{T}(undef, nrow, k)  # centroids

    return (batch_rand_idx = batch_rand_idx, centroids_new = centroids_new,
            labels = labels, sum_of_squares = sum_of_squares)
end
