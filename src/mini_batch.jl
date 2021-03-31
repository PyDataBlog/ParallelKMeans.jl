"""
    MiniBatch(b::Int)
    `b` represents the size of the batch which should be sampled.

    Sculley et al. 2007 Mini batch k-means algorithm implementation.

```julia
X = rand(30, 100_000)  # 100_000 random points in 30 dimensions

kmeans(MiniBatch(100), X, 3)  # 3 clusters, MiniBatch algorithm with 100 batch samples at each iteration
```
"""
struct MiniBatch <: AbstractKMeansAlg
    b::Int  # batch size
end


MiniBatch() = MiniBatch(100)

function kmeans!(alg::MiniBatch, containers, X, k,
                 weights = nothing, metric = Euclidean(); n_threads = Threads.nthreads(),
                 k_init = "k-means++", init = nothing, max_iters = 300,
                 tol = eltype(X)(1e-6), max_no_improvement = 10, verbose = false, rng = Random.GLOBAL_RNG)

    # Get the type and dimensions of design matrix, X - (Step 1)
    T = eltype(X)
    nrow, ncol = size(X)

    # Initiate cluster centers - (Step 2) in paper
    centroids = isnothing(init) ? smart_init(X, k, n_threads, weights, rng, init = k_init).centroids : deepcopy(init)

    # Initialize counter for the no. of data in each cluster - (Step 3) in paper
    N = zeros(T, k)

    # Initialize nearest centers for both batch and whole dataset labels
    converged = false
    niters = 0
    counter = 0
    J_previous = zero(T)
    J = zero(T)
    totalcost = zero(T)

    # Main Steps. Batch update centroids until convergence
    while niters <= max_iters  # Step 4 in paper

        # b examples picked randomly from X (Step 5 in paper)
        batch_rand_idx = isnothing(weights) ? rand(rng, 1:ncol, alg.b) : wsample(rng, 1:ncol, weights, alg.b)

        # Cache/label the batch samples nearest to the centers (Step 6 & 7)
        @inbounds for i in batch_rand_idx
            min_dist = distance(metric, X, centroids, i, 1)
            label = 1

            for j in 2:size(centroids, 2)
                dist = distance(metric, X, centroids, i, j)
                label = dist < min_dist ? j : label
                min_dist = dist < min_dist ? dist : min_dist
            end

            containers.labels[i] = label
        end

        # Batch gradient step
        @inbounds for j in batch_rand_idx  # iterate over examples (Step 9)

            # Get cached center/label for this x  => (Step 10)
            label = containers.labels[j]
            
            # Update per-center counts
            N[label] += isnothing(weights) ? 1 : weights[j]  # (Step 11)

            # Get per-center learning rate (Step 12)
            lr = 1 / N[label]

            # Take gradient step (Step 13) # TODO: Replace with faster loop?
            @views centroids[:, label] .= (1 - lr) .* centroids[:, label] .+ (lr .* X[:, j])
        end

        # Reassign all labels based on new centres generated from the latest sample
        containers.labels .= reassign_labels(X, metric, containers.labels, centroids)

        # Calculate cost on whole dataset after reassignment and check for convergence
        @parallelize 1 ncol sum_of_squares(containers, X, containers.labels, centroids, weights, metric)  
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
                containers.labels .= reassign_labels(X, metric, containers.labels, centroids)

                # Compute totalcost for the complete dataset
                @parallelize 1 ncol sum_of_squares(containers, X, containers.labels, centroids, weights, metric)
                totalcost = sum(containers.sum_of_squares)
                break
            end
        else
            counter = 0
        end

        # Warn users if model doesn't converge at max iterations
        if (niters > max_iters) & (!converged)

            println("Clustering model failed to converge. Labelling data with latest centroids.")
            containers.labels = reassign_labels(X, metric, containers.labels, centroids)

            # Compute totalcost for unconverged model
            @parallelize 1 ncol sum_of_squares(containers, X, containers.labels, centroids, weights, metric)
            totalcost = sum(containers.sum_of_squares)
            break
        end

        J_previous = J
        niters += 1
    end

    # Push learned artifacts to KmeansResult
    return KmeansResult(centroids, containers.labels, T[], Int[], T[], totalcost, niters, converged)
end


function reassign_labels(DMatrix, metric, labels, centres)
    @inbounds for i in axes(DMatrix, 2)
        min_dist = distance(metric, DMatrix, centres, i, 1)
        label = 1

        for j in 2:size(centres, 2)
            dist = distance(metric, DMatrix, centres, i, j)
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
- `centroids_cnt` - container which holds number of points for each centroid
- `labels` - vector which holds labels of corresponding points
- `sum_of_squares` - vector which holds the sum of squares values for each thread
"""
function create_containers(::MiniBatch, X, k, nrow, ncol, n_threads)
    # Initiate placeholders to avoid allocations
    T = eltype(X) 
    centroids_new = Matrix{T}(undef, nrow, k)  # main centroids
    centroids_cnt = Vector{T}(undef, k)  # centroids counter
    labels = Vector{Int}(undef, ncol)  # labels vector
    sum_of_squares = Vector{T}(undef, 1)  # total_sum_calculation

    return (centroids_new = centroids_new, centroids_cnt = centroids_cnt,
            labels = labels, sum_of_squares = sum_of_squares)
end
