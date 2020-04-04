"""
    LightElkan <: AbstractKMeansAlg

Simplified version of Elkan algorithm for k-means calculation. This algorithm
gives the same results as basic Lloyd algorithm, but improve in speed by omitting
unnecessary calculations. In this implementation there are two conditions applied
to accelerate calculations

- if point is sufficiently close to it's centroid, i.e. distance to the centroid is smaller than
half minimum distance from centroid to all other centroid. In this scenario point immediately get
label of closest centroid.
- if during calculation of new label distance from the point to current centroid is less than
half of the distance from centroid to any other centroid, then distance from the point to
other centroid is not calculated.

One has to take into account, that LightElkan algorithm has an overhead of the calculation
k x k matrix of centroids distances, so for tasks with no apparent cluster structure may perform
worser than basic LLoyd algorithm.
"""
struct LightElkan <: AbstractKMeansAlg end


"""
    create_containers(::LightElkan, k, nrow, ncol, n_threads)

Internal function for the creation of all necessary intermidiate structures.

- `new_centroids` - container which holds new positions of centroids
- `centroids_cnt` - container which holds number of points for each centroid
- `labels` - vector which holds labels of corresponding points
- `centroids_dist` - symmetric matrix k x k which holds weighted distances between centroids
"""
function create_containers(alg::LightElkan, k, nrow, ncol, n_threads)
    new_centroids = Vector{Array{Float64, 2}}(undef, n_threads)
    centroids_cnt = Vector{Vector{Int}}(undef, n_threads)

    for i in 1:n_threads
        new_centroids[i] = Array{Float64, 2}(undef, nrow, k)
        centroids_cnt[i] = Vector{Int}(undef, k)
    end

    labels = zeros(Int, ncol)

    centroids_dist = Matrix{Float64}(undef, k, k)

    return (new_centroids = new_centroids, centroids_cnt = centroids_cnt,
            labels = labels, centroids_dist = centroids_dist)
end


"""
    update_containers!(containers, ::LightElkan, centroids, n_threads)

Internal function for the `LightElkan` algorithm which updates distances
between centroids. These distances are presented as symmetric matrix,
on diagonal is written minimal distance from current centroid to all other.
All distances are weighted with the factor 0.25 in order to simplify following
update_centroids calculations.
"""
function update_containers!(containers, ::LightElkan, centroids, n_threads)
    # unpack containers for easier manipulations
    centroids_dist = containers.centroids_dist

    k = size(centroids_dist, 1) # number of clusters
    @inbounds for j in axes(centroids_dist, 2)
        min_dist = Inf
        for i in j + 1:k
            d = 0.0
            for m in axes(centroids, 1)
                d += (centroids[m, i] - centroids[m, j])^2
            end
            centroids_dist[i, j] = d
            centroids_dist[j, i] = d
            min_dist = min_dist < d ? min_dist : d
        end
        for i in 1:j - 1
            min_dist = min_dist < centroids_dist[j, i] ? min_dist : centroids_dist[j, i]
        end
        centroids_dist[j, j] = min_dist
    end

    # TODO: oh, one should be careful here. inequality holds for eucledian metrics
    # not square eucledian. So, for Lp norm it should be something like
    # centroids_dist = 0.5^p. Should check one more time original paper
    centroids_dist .*= 0.25

    return centroids_dist
end

"""
    chunk_update_centroids!(centroids, containers, ::AbstractKMeansAlg, design_matrix, r, idx)

Internal function which calculates single centroids update for data chunk.

Argument `idx` denotes number of the thread used, if it is equals 0 it means, that we are in single
thread mode.
"""
function chunk_update_centroids!(centroids, containers, ::LightElkan,
    design_matrix, r, idx)

    # unpack containers for easier manipulations
    new_centroids = containers.new_centroids[idx]
    centroids_cnt = containers.centroids_cnt[idx]
    centroids_dist = containers.centroids_dist
    labels = containers.labels

    new_centroids .= 0.0
    centroids_cnt .= 0
    J = 0.0
    @inbounds for i in r
        # calculate distance to the previous center
        label = labels[i] > 0 ? labels[i] : 1
        last_label = label
        distance = 0.0
        for j in axes(design_matrix, 1)
            distance += (design_matrix[j, i] - centroids[j, label])^2
        end

        min_distance = distance

        # we can optimize in two ways
        # if point is close (less then centroids_dist[i, i]) to the center then there is no need to recalculate it
        # if it's not close, then we can skip some of centers if the center is too far away from
        # current point (Elkan triangular inequality)
        if min_distance > centroids_dist[label, label]
            for k in axes(centroids, 2)
                k == last_label && continue
                # triangular inequality
                centroids_dist[k, label] > min_distance && continue
                distance = 0.0
                for j in axes(design_matrix, 1)
                    # TODO: we can break this calculation if distance already larger than
                    # min_distance
                    distance += (design_matrix[j, i] - centroids[j, k])^2
                end
                label = min_distance > distance ? k : label
                min_distance = min_distance > distance ? distance : min_distance
            end
        end

        labels[i] = label
        centroids_cnt[label] += 1
        for j in axes(design_matrix, 1)
            new_centroids[j, label] += design_matrix[j, i]
        end
        J += min_distance
    end

    return J
end
