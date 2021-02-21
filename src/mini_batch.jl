"""
    MiniBatch(b::Int)

"""
struct MiniBatch <: AbstractKMeansAlg
    b::Int  # batch size
end


MiniBatch() = MiniBatch(100)

function kmeans!(alg::MiniBatch, X, k;
                 weights = nothing, metric = Euclidean(), n_threads = Threads.nthreads(),
                 k_init = "k-means++", init = nothing, max_iters = 300,
<<<<<<< HEAD
                 tol = 0, max_no_improvement = 10, verbose = false, rng = Random.GLOBAL_RNG)
=======
                 tol = 0, verbose = false, rng = Random.GLOBAL_RNG)
>>>>>>> 83de57e (MiniBatch algorithm)

    # Step 1. Select sample from X as specified by batch_size and weights
    nrow, ncol = size(X)  # n_features, m_examples
    T = eltype(X)

    # Step 2. Initiate cluster centers from the initial sample
    centroids = isnothing(init) ? smart_init(X, k, n_threads, weights, rng, init = k_init).centroids : deepcopy(init)

    # initialize nearest centers
    labels = Vector{Int}(undef, alg.b)

    # initialize no. of data in each cluster
    N = zeros(T, k)

    # TODO: Initialise controid counting caches.(Dimensions right?)
    #centroids_cnt = Vector{Int}(undef, k)

    converged = false
    niters = 0
    J_previous = zero(T)
    J = zero(T)

    # TODO: Main Steps. Batch update centroids until convergence
    while niters <= max_iters
<<<<<<< HEAD
        counter = 0

=======
>>>>>>> 83de57e (MiniBatch algorithm)
        # b examples picked randomly from X (Stage 5 in paper)
        batch_rand_idx = isnothing(weights) ? rand(rng, 1:ncol, alg.b) : wsample(rng, 1:ncol, weights, alg.b)
        batch_sample = X[:, batch_rand_idx]

        # Cache/label the batch samples nearest to the centers
        @inbounds for i in axes(batch_sample, 2)
            min_dist = distance(metric, batch_sample, centroids, i, 1)
            label = 1

            for j in 2:size(centroids, 2)
                dist = distance(metric, batch_sample, centroids, i, j)
                label = dist < min_dist ? j : label
                min_dist = dist < min_dist ? dist : min_dist
            end

            labels[i] = label
        end

        # TODO: Batch gradient step
        for j in axes(batch_sample, 2)  # iterate over examples

            # Get cached center/labrel for this x  => labels[j]
            label = labels[j]
            # Update per-center counts
            N[label] += isnothing(weights) ? 1 : weights[j]  # verify

            # Get per-center learning rate
            lr = 1 / N[label]

            # Take gradient step # TODO: Replace with an allocation-less loop.
            centroids[:, label] .= (1 - lr) .* centroids[:, label] .+ (lr .* batch_sample[:, j])
        end

        # TODO: Calculate cost and check for convergence
        J = sum_of_squares(batch_sample, labels, centroids)  # just a placeholder for now

        if verbose
            # Show progress and terminate if J stopped decreasing.
            println("Iteration $niters: Jclust = $J")
        end

        # TODO: Check for early stopping convergence
        if (niters > 1) & abs(J - J_previous)
            counter += 1

            # Declare convergence if max_no_improvement criterion is met
            if counter >= max_no_improvement
                converged = true
                break
            end

        end

        J_previous = J
        niters += 1
    end

    return centroids, niters, converged, labels, J  # TODO: push learned artifacts to KmeansResult
    #return KmeansResult(centroids, containers.labels, T[], Int[], T[], totalcost, niters, converged)
end

# TODO: Only being used to test generic implementation. Get rid off after! Unverified func!
function sum_of_squares(x, labels, centre)
    s = 0.0

    for i in axes(x, 2)
        for j in axes(x, 1)
            s += (x[j, i] - centre[j, labels[i]])^2
        end
    end
    return s
end
