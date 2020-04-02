struct Hamerly <: AbstractKMeansAlg end

function kmeans(alg::Hamerly, design_matrix, k;
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = 1e-6, verbose = true, init = nothing)
    nrow, ncol = size(design_matrix)
    containers = create_containers(alg, k, nrow, ncol, n_threads)

    return kmeans!(alg, containers, design_matrix, k, n_threads = n_threads,
                    k_init = k_init, max_iters = max_iters, tol = tol,
                    verbose = verbose, init = init)
end

function kmeans!(alg::Hamerly, containers, design_matrix, k;
                n_threads = Threads.nthreads(),
                k_init = "k-means++", max_iters = 300,
                tol = 1e-6, verbose = true, init = nothing)
    nrow, ncol = size(design_matrix)
    centroids = init == nothing ? smart_init(design_matrix, k, n_threads, init=k_init).centroids : deepcopy(init)

    initialize!(alg, containers, centroids, design_matrix, n_threads)

    converged = false
    niters = 1
    J_previous = 0.0

    # Update centroids & labels with closest members until convergence

    while niters <= max_iters
        update_containers!(containers, alg, centroids, n_threads)
        update_centroids!(centroids, containers, alg, design_matrix, n_threads)
        J = sum(containers.ub)
        move_centers!(centroids, containers, alg)
        update_bounds!(containers, n_threads)

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

function collect_containers(alg::Hamerly, containers, n_threads)
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

function create_containers(alg::Hamerly, k, nrow, ncol, n_threads)
    lng = n_threads + 1
    centroids_new = Vector{Array{Float64,2}}(undef, lng)
    centroids_cnt = Vector{Vector{Int}}(undef, lng)

    for i = 1:lng
        centroids_new[i] = zeros(nrow, k)
        centroids_cnt[i] = zeros(k)
    end

    # Upper bound to the closest center
    ub = Vector{Float64}(undef, ncol)

    # lower bound to the second closest center
    lb = Vector{Float64}(undef, ncol)

    labels = zeros(Int, ncol)

    # distance that centroid moved
    p = Vector{Float64}(undef, k)

    # distance from the center to the closest other center
    s = Vector{Float64}(undef, k)

    return (
        centroids_new = centroids_new,
        centroids_cnt = centroids_cnt,
        labels = labels,
        ub = ub,
        lb = lb,
        p = p,
        s = s,
    )
end

function initialize!(alg::Hamerly, containers, centroids, design_matrix, n_threads)
    ncol = size(design_matrix, 2)

    if n_threads == 1
        r = axes(design_matrix, 2)
        chunk_initialize!(alg, containers, centroids, design_matrix, r, 1)
    else
        ranges = splitter(ncol, n_threads)

        waiting_list = Vector{Task}(undef, n_threads - 1)

        for i in 1:n_threads - 1
            waiting_list[i] = @spawn chunk_initialize!(alg, containers, centroids,
                design_matrix, ranges[i], i + 1)
        end

        chunk_initialize!(alg, containers, centroids, design_matrix, ranges[end], 1)

        wait.(waiting_list)
    end
end

function chunk_initialize!(alg::Hamerly, containers, centroids, design_matrix, r, idx)
    centroids_cnt = containers.centroids_cnt[idx]
    centroids_new = containers.centroids_new[idx]

    @inbounds for i in r
        label = point_all_centers!(containers, centroids, design_matrix, i)
        centroids_cnt[label] += 1
        for j in axes(design_matrix, 1)
            centroids_new[j, label] += design_matrix[j, i]
        end
    end
end

function update_containers!(containers, ::Hamerly, centroids, n_threads)
    s = containers.s
    s .= Inf
    @inbounds for i in axes(centroids, 2)
        for j in i+1:size(centroids, 2)
            d = distance(centroids, centroids, i, j)
            d = 0.25*d
            s[i] = s[i] > d ? d : s[i]
            s[j] = s[j] > d ? d : s[j]
        end
    end
end

function update_centroids!(centroids, containers, alg::Hamerly, design_matrix, n_threads)

    if n_threads == 1
        r = axes(design_matrix, 2)
        chunk_update_centroids!(centroids, containers, alg, design_matrix, r, 1)
    else
        ncol = size(design_matrix, 2)
        ranges = splitter(ncol, n_threads)

        waiting_list = Vector{Task}(undef, n_threads - 1)

        for i in 1:length(ranges) - 1
            waiting_list[i] = @spawn chunk_update_centroids!(centroids, containers,
                alg, design_matrix, ranges[i], i)
        end

        chunk_update_centroids!(centroids, containers, alg, design_matrix, ranges[end], n_threads)

        wait.(waiting_list)

    end

    collect_containers(alg, containers, n_threads)
end

function chunk_update_centroids!(
    centroids,
    containers,
    alg::Hamerly,
    design_matrix,
    r,
    idx,
)

    # unpack containers for easier manipulations
    centroids_new = containers.centroids_new[idx]
    centroids_cnt = containers.centroids_cnt[idx]
    labels = containers.labels
    s = containers.s
    lb = containers.lb
    ub = containers.ub

    @inbounds for i in r
        # m ← max(s(a(i))/2, l(i))
        m = max(s[labels[i]], lb[i])
        # first bound test
        if ub[i] > m
            # tighten upper bound
            label = labels[i]
            ub[i] = distance(design_matrix, centroids, i, label)
            # second bound test
            if ub[i] > m
                label_new = point_all_centers!(containers, centroids, design_matrix, i)
                if label != label_new
                    labels[i] = label_new
                    centroids_cnt[label_new] += 1
                    centroids_cnt[label] -= 1
                    for j in axes(design_matrix, 1)
                        centroids_new[j, label_new] += design_matrix[j, i]
                        centroids_new[j, label] -= design_matrix[j, i]
                    end
                end
            end
        end
    end
end

function point_all_centers!(containers, centroids, design_matrix, i)
    ub = containers.ub
    lb = containers.lb
    labels = containers.labels

    min_distance = Inf
    min_distance2 = Inf
    label = 1
    @inbounds for k in axes(centroids, 2)
        dist = distance(design_matrix, centroids, i, k)
        if min_distance > dist
            label = k
            min_distance2 = min_distance
            min_distance = dist
        elseif min_distance2 > dist
            min_distance2 = dist
        end
    end

    ub[i] = min_distance
    lb[i] = min_distance2
    labels[i] = label

    return label
end

function move_centers!(centroids, containers, ::Hamerly)
    centroids_new = containers.centroids_new[end]
    p = containers.p

    @inbounds for i in axes(centroids, 2)
        d = 0.0
        for j in axes(centroids, 1)
            d += (centroids[j, i] - centroids_new[j, i])^2
            centroids[j, i] = centroids_new[j, i]
        end
        p[i] = d
    end
end

function update_bounds!(containers, n_threads)
    p = containers.p

    r1, r2 = double_argmax(p)
    pr1 = p[r1]
    pr2 = p[r2]

    if n_threads == 1
        r = axes(containers.ub, 1)
        chunk_update_bounds!(containers, r, r1, r2, pr1, pr2)
    else
        ncol = length(containers.ub)
        ranges = splitter(ncol, n_threads)

        waiting_list = Vector{Task}(undef, n_threads - 1)

        for i in 1:n_threads - 1
            waiting_list[i] = @spawn chunk_update_bounds!(containers, ranges[i], r1, r2, pr1, pr2)
        end

        chunk_update_bounds!(containers, ranges[end], r1, r2, pr1, pr2)

        for i in 1:n_threads - 1
            wait(waiting_list[i])
        end
    end
end

function chunk_update_bounds!(containers, r, r1, r2, pr1, pr2)
    p = containers.p
    ub = containers.ub
    lb = containers.lb
    labels = containers.labels

    @inbounds for i in r
        label = labels[i]
        ub[i] += 2*sqrt(abs(ub[i] * p[label])) + p[label]
        if r1 == label
            lb[i] += pr2 - 2*sqrt(abs(pr2*lb[i]))
        else
            lb[i] += pr1 - 2*sqrt(abs(pr1*lb[i]))
        end
    end
end

function double_argmax(p)
    r1, r2 = 1, 1
    d1 = p[1]
    d2 = -1.0
    for i in 2:length(p)
        if p[i] > d1
            r2 = r1
            r1 = i
            d2 = d1
            d1 = p[i]
        elseif p[i] > d2
            d2 = p[i]
            r2 = i
        end
    end

    r1, r2
end

"""
    distance(X1, X2, i1, i2)

Allocation less calculation of square eucledean distance between vectors X1[:, i1] and X2[:, i2]
"""
function distance(X1, X2, i1, i2)
    d = 0.0
    @inbounds for i in axes(X1, 1)
        d += (X1[i, i1] - X2[i, i2])^2
    end

    return d
end
