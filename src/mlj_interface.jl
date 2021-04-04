# Expose all instances of user specified structs and package artifcats.
const MMI = MLJModelInterface

const ParallelKMeans_Desc = "Parallel & lightning fast implementation of all available variants of the KMeans clustering algorithm
                             in native Julia. Compatible with Julia 1.3+"

# availalbe variants for reference
const MLJDICT = Dict(:Lloyd => Lloyd(),
                     :Hamerly => Hamerly(),
                     :Elkan => Elkan(),
					 :Yinyang => Yinyang(),
					 :Coreset => Coreset(),
					 :阴阳 => Coreset(), 
                     :MiniBatch => MiniBatch())

####
#### MODEL DEFINITION
####
"""
    ParallelKMeans model constructed by the user.
    See also the [package documentation](https://pydatablog.github.io/ParallelKMeans.jl/stable).
"""
mutable struct KMeans <: MMI.Unsupervised
    algo::Union{Symbol, AbstractKMeansAlg}
    k_init::String
    k::Int
    tol::Float64
    max_iters::Int
    copy::Bool
    threads::Int
    rng::Union{AbstractRNG, Int}
	weights
    init
end


function KMeans(; algo = :Hamerly, k_init = "k-means++",
                k = 3, tol = 1e-6, max_iters = 300, copy = true,
                threads = Threads.nthreads(), init = nothing,
				rng = Random.GLOBAL_RNG, weights = nothing)

    model   = KMeans(algo, k_init, k, tol, max_iters, copy, threads, rng, weights, init)
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end


function MMI.clean!(m::KMeans)
	warning = String[]

	m.algo = clean_algo(m.algo, warning)

    if !(m.k_init ∈ ["k-means++", "random"])
        push!(warning, "Only \"k-means++\" or \"random\" seeding algorithms are supported. Defaulting to k-means++ seeding.")
        m.k_init = "kmeans++"
	end

    if m.k < 1
        push!(warning, "Number of clusters must be greater than 0. Defaulting to 3 clusters.")
        m.k = 3
	end

    if !(m.tol < 1.0)
        push!(warning, "Tolerance level must be less than 1. Defaulting to tol of 1e-6.")
        m.tol = 1e-6
	end

    if !(m.max_iters > 0)
        push!(warning, "Number of permitted iterations must be greater than 0. Defaulting to 300 iterations.")
        m.max_iters = 300
	end

    if !(m.threads > 0)
        push!(warning, "Number of threads must be at least 1. Defaulting to all threads available.")
        m.threads = Threads.nthreads()
	end

    return join(warning, "\n")
end


####
#### FIT FUNCTION
####
"""
    Fit the specified ParallelKMeans model constructed by the user.

    See also the [package documentation](https://pydatablog.github.io/ParallelKMeans.jl/stable).
"""
function MMI.fit(m::KMeans, verbosity::Int, X)
    # convert tabular input data into the matrix model expects. Column assumed as features so input data is permuted
    if !m.copy
        # permutes dimensions of input table without copying and pass to model
        DMatrix = convert(Array{Float64, 2}, MMI.matrix(X)')
    else
        # permutes dimensions of input table as a column major matrix from a copy of the data
        DMatrix = convert(Array{Float64, 2}, MMI.matrix(X, transpose=true))
    end

	# setup rng
	rng = get_rng(m.rng)

	if !isnothing(m.weights) && (size(DMatrix, 2) != length(m.weights))
		@warn "Size mismatch, number of points in X $(size(DMatrix, 2)) not equal weights length $(length(m.weights)). Weights parameter ignored."
		weights = nothing
	else

		weights = m.weights
	end

    # fit model and get results
    verbose = verbosity > 0  # Display fitting operations if verbosity > 0
    result = ParallelKMeans.kmeans(m.algo, DMatrix, m.k;
                                      n_threads = m.threads, k_init = m.k_init,
                                      max_iters = m.max_iters, tol = m.tol, init = m.init,
                                      rng = rng, verbose = verbose, weights = weights)

    cluster_labels = MMI.categorical(1:m.k)
    fitresult = (centers = result.centers, labels = cluster_labels, converged = result.converged)
    cache = nothing

    report = (cluster_centers=result.centers, iterations=result.iterations,
              totalcost=result.totalcost, assignments=result.assignments, labels=cluster_labels)


    # Warn users about non convergence
    if verbose & (!fitresult.converged)
        @warn "Specified model failed to converge."
    end

    return (fitresult, cache, report)
end


function MMI.fitted_params(model::KMeans, fitresult)
    # Centroids
    return (cluster_centers = fitresult.centers, )
end


####
#### PREDICT FUNCTION
####

function MMI.transform(m::KMeans, fitresult, Xnew)
    # transform new data using the fitted centroids.

    if !m.copy
        # permutes dimensions of input table without copying and pass to model
        DMatrix = convert(Array{Float64, 2}, MMI.matrix(Xnew)')
    else
        # permutes dimensions of input table as a column major matrix from a copy of the data
        DMatrix = convert(Array{Float64, 2}, MMI.matrix(Xnew, transpose=true))
    end

    # Warn users if fitresult is from a `non-converged` fit
    if !fitresult.converged
        @warn "Failed to converge. Using last assignments to make transformations."
    end

    # use centroid matrix to assign clusters for new data
    distances = Distances.pairwise(Distances.SqEuclidean(), DMatrix, fitresult.centers; dims=2)
    #preds = argmin.(eachrow(distances))
    return MMI.table(distances, prototype=Xnew)
end


function MMI.predict(m::KMeans, fitresult, Xnew)
    locations, cluster_labels, _ = fitresult

    Xarray = MMI.matrix(Xnew)
    # TODO: Switch to non allocation method.
    (n, p), k = size(Xarray), m.k

    pred = zeros(Int, n)
    @inbounds for i ∈ 1:n
        minv = Inf
        for j ∈ 1:k
            curv    = Distances.evaluate(Distances.Euclidean(), view(Xarray, i, :), view(locations, :, j))
            P       = curv < minv
            pred[i] =    j * P + pred[i] * !P # if P is true --> j
            minv    = curv * P +    minv * !P # if P is true --> curvalue
        end
    end
    return cluster_labels[pred]
end

####
#### METADATA
####

# Metadata for the package and for each of the model interfaces
MMI.metadata_pkg.(KMeans,
    name        = "ParallelKMeans",
    uuid        = "42b8e9d4-006b-409a-8472-7f34b3fb58af",
    url         = "https://github.com/PyDataBlog/ParallelKMeans.jl",
    julia       = true,
    license     = "MIT",
    is_wrapper  = false)


# Metadata for ParaKMeans model interface
MMI.metadata_model(KMeans,
    input   = MMI.Table(MMI.Continuous),
    output  = MMI.Table(MMI.Continuous),
    target  =  AbstractArray{<:MMI.Multiclass},
    weights = false,
    descr   = ParallelKMeans_Desc,
	path	= "ParallelKMeans.KMeans")

####
#### Auxiliary functions
####

get_rng(rng::Int) = MersenneTwister(rng)
get_rng(rng) = rng

clean_algo(algo::AbstractKMeansAlg, warning) = algo
function clean_algo(algo::Symbol, warning)
	if !(algo ∈ keys(MLJDICT))
		push!(warning, "Unsupported KMeans variant. Defaulting to Hamerly algorithm.")
		return MLJDICT[:Hamerly]
	else
		return MLJDICT[algo]
	end
end
