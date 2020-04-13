# Expose all instances of user specified structs and package artifcats.
const ParallelKMeans_Desc = "Parallel & lightning fast implementation of all available variants of the KMeans clustering algorithm
                             in native Julia. Compatible with Julia 1.3+"

# availalbe variants for reference
const MLJDICT = Dict(:Lloyd => Lloyd(),
                     :Hamerly => Hamerly(),
                     :Elkan => Elkan())

####
#### MODEL DEFINITION
####

mutable struct KMeans <: MMI.Unsupervised
    algo::Symbol
    k_init::String
    k::Int
    tol::Float64
    max_iters::Int
    copy::Bool
    threads::Int
    verbosity::Int
    init
end


function KMeans(; algo=:Hamerly, k_init="k-means++",
                k=3, tol=1e-6, max_iters=300, copy=true,
                threads=Threads.nthreads(), verbosity=0, init=nothing)

    model   = KMeans(algo, k_init, k, tol, max_iters, copy, threads, verbosity, init)
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end


function MMI.clean!(m::KMeans)
    warning = String[]

    if !(m.algo ∈ keys(MLJDICT))
        push!(warning, "Unsupported KMeans variant. Defaulting to Hamerly algorithm.")
        m.algo = :Hamerly
	end

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

    if !(m.verbosity ∈ (0, 1))
        push!(warning, "Verbosity must be either 0 (no info) or 1 (info requested). Defaulting to 1.")
        m.verbosity = 1
    end

    return join(warning, "\n")
end


####
#### FIT FUNCTION
####
"""
    Fit the specified ParaKMeans model constructed by the user.

    See also the [package documentation](https://pydatablog.github.io/ParallelKMeans.jl/stable).
"""
function MMI.fit(m::KMeans, X)
    # convert tabular input data into the matrix model expects. Column assumed as features so input data is permuted
    if !m.copy
        # permutes dimensions of input table without copying and pass to model
        DMatrix = convert(Array{Float64, 2}, MMI.matrix(X)')
    else
        # permutes dimensions of input table as a column major matrix from a copy of the data
        DMatrix = convert(Array{Float64, 2}, MMI.matrix(X, transpose=true))
    end

    # lookup available algorithms
    algo = MLJDICT[m.algo]  # select algo

    # fit model and get results
    verbose = m.verbosity != 0
    fitresult = ParallelKMeans.kmeans(algo, DMatrix, m.k;
                                      n_threads = m.threads, k_init=m.k_init,
                                      max_iters=m.max_iters, tol=m.tol, init=m.init,
                                      verbose=verbose)
    cache = nothing
    report = (cluster_centers=fitresult.centers, iterations=fitresult.iterations,
              converged=fitresult.converged, totalcost=fitresult.totalcost,
              labels=fitresult.assignments)

    return (fitresult, cache, report)
end


function MMI.fitted_params(model::KMeans, fitresult)
    # extract what's relevant from `fitresult`
    results, _, _ = fitresult  # unpack fitresult
    centers = results.centers
    converged = results.converged
    iters = results.iterations
    totalcost = results.totalcost

    # then return as a NamedTuple
    return (cluster_centers = centers, totalcost = totalcost,
            iterations = iters, converged = converged)
end


####
#### PREDICT FUNCTION
####

function MMI.transform(m::KMeans, fitresult, Xnew)
    # make predictions/assignments using the learned centroids

    if !m.copy
        # permutes dimensions of input table without copying and pass to model
        DMatrix = convert(Array{Float64, 2}, MMI.matrix(Xnew)')
    else
        # permutes dimensions of input table as a column major matrix from a copy of the data
        DMatrix = convert(Array{Float64, 2}, MMI.matrix(Xnew, transpose=true))
    end

    # Warn users if fitresult is from a `non-converged` fit
    if !fitresult[end].converged
        @warn "Failed to converged. Using last assignments to make transformations."
    end

    # results from fitted model
    results = fitresult[1]

    # use centroid matrix to assign clusters for new data
    centroids = results.centers
    distances = Distances.pairwise(Distances.SqEuclidean(), DMatrix, centroids; dims=2)
    preds = argmin.(eachrow(distances))
    return MMI.table(reshape(preds, :, 1), prototype=Xnew)
end


####
#### METADATA
####

# TODO 4: metadata for the package and for each of the model interfaces
MMI.metadata_pkg.(KMeans,
    name = "ParallelKMeans",
    uuid = "42b8e9d4-006b-409a-8472-7f34b3fb58af",
    url  = "https://github.com/PyDataBlog/ParallelKMeans.jl",
    julia = true,
    license = "MIT",
    is_wrapper = false)


# Metadata for ParaKMeans model interface
MMI.metadata_model(KMeans,
    input   = MMI.Table(MMI.Continuous),
    output  = MMI.Table(MMI.Count),
    weights = false,
    descr   = ParallelKMeans_Desc,
	path	= "ParallelKMeans.KMeans")
