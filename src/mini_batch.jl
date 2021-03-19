"""
    MiniBatch(b::Int)

    Sculley et al. 2007 Mini batch k-means algorithm implementation.
"""
struct MiniBatch <: AbstractKMeansAlg
    b::Int  # batch size
end


MiniBatch() = MiniBatch(100)

function kmeans!(alg::MiniBatch, X, k;
                 weights = nothing, metric = Euclidean(), n_threads = Threads.nthreads(),
                 k_init = "k-means++", init = nothing, max_iters = 300,
                 tol = eltype(X)(1e-6), max_no_improvement = 10, verbose = false, rng = Random.GLOBAL_RNG)

    # Get the type and dimensions of design matrix, X
    T = eltype(X)
    nrow, ncol = size(X)

    # Initiate cluster centers - (Step 2) in paper
    centroids = isnothing(init) ? smart_init(X, k, n_threads, weights, rng, init = k_init).centroids : deepcopy(init)

    # Initialize counter for the no. of data in each cluster - (Step 3) in paper
    N = zeros(T, k)

    # Initialize nearest centers
    labels = Vector{Int}(undef, alg.b)
    final_labels = Vector{Int}(undef, ncol)

    converged = false
    niters = 0
    counter = 0
    J_previous = zero(T)
    J = zero(T)

    # TODO: Main Steps. Batch update centroids until convergence
    while niters <= max_iters

        # b examples picked randomly from X (Step 5 in paper)
        batch_rand_idx = isnothing(weights) ? rand(rng, 1:ncol, alg.b) : wsample(rng, 1:ncol, weights, alg.b)
        batch_sample = X[:, batch_rand_idx]

        # Cache/label the batch samples nearest to the centers (Step 6 & 7)
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
        for j in axes(batch_sample, 2)  # iterate over examples (Step 9)

            # Get cached center/label for this x  => labels[j] (Step 10)
            label = labels[j]
            # Update per-center counts
            N[label] += isnothing(weights) ? 1 : weights[j]  # verify (Step 11)

            # Get per-center learning rate (Step 12)
            lr = 1 / N[label]

            # Take gradient step (Step 13) # TODO: Replace with an allocation-less loop.
            centroids[:, label] .= (1 - lr) .* centroids[:, label] .+ (lr .* batch_sample[:, j])
        end

        # TODO: Calculate cost and check for convergence
        J = sum_of_squares(batch_sample, labels, centroids)  # just a placeholder for now

        if verbose
            # Show progress and terminate if J stopped decreasing.
            println("Iteration $niters: Jclust = $J")
        end

        # TODO: Check for early stopping convergence
        if (niters > 1) & (abs(J - J_previous) < (tol * J))
            counter += 1

            # Declare convergence if max_no_improvement criterion is met
            if counter >= max_no_improvement
                converged = true
                # TODO: Compute label assignment for the complete dataset
                @inbounds for i in axes(X, 2)
                    min_dist = distance(metric, X, centroids, i, 1)
                    label = 1

                    for j in 2:size(centroids, 2)
                        dist = distance(metric, X, centroids, i, j)
                        label = dist < min_dist ? j : label
                        min_dist = dist < min_dist ? dist : min_dist
                    end

                    final_labels[i] = label
                end
                # TODO: Compute totalcost for the complete dataset
                J = sum_of_squares(X, final_labels, centroids)  # just a placeholder for now
                break
            end
        else
            counter = 0

        end

        J_previous = J
        niters += 1
    end

    return centroids, niters, converged, final_labels, J  # TODO: push learned artifacts to KmeansResult
    #return KmeansResult(centroids, containers.labels, T[], Int[], T[], totalcost, niters, converged)
end

# TODO: Only being used to test generic implementation. Get rid off after!
function sum_of_squares(x, labels, centre)
    s = 0.0

    for i in axes(x, 2)
        for j in axes(x, 1)
            s += (x[j, i] - centre[j, labels[i]])^2
        end
    end
    return s
end
