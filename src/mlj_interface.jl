# TODO 1: a using MLJModelInterface or import MLJModelInterface statement
# Expose all instances of user specified structs and package artifcats.
const ParallelKMeans_Desc = "Parallel & lightning fast implementation of all variants of the KMeans clustering algorithm in native Julia."

# availalbe variants for reference
const MLJDICT = Dict(:Lloyd => Lloyd(),
                     :Hamerly => Hamerly(),
                     :LightElkan => LightElkan())

####
#### MODEL DEFINITION
####
# TODO 2: MLJ-compatible model types and constructors

mutable struct KMeans <: MLJModelInterface.Unsupervised
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


function KMeans(; algo=:Lloyd, k_init="k-means++",
                k=3, tol=1e-6, max_iters=300, copy=true,
                threads=Threads.nthreads(), verbosity=0, init=nothing)

    model   = KMeans(algo, k_init, k, tol, max_iters, copy, threads, verbosity, init)
    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end


function MLJModelInterface.clean!(m::KMeans)
    warning = ""

    if !(m.algo ∈ keys(MLJDICT))
        warning *= "Unsuppored algorithm supplied. Defauting to KMeans++ seeding algorithm."
        m.algo = :Lloyd

    elseif m.k_init != "k-means++"
        warning *= "Only `k-means++` or random seeding algorithms are supported. Defaulting to random seeding."
        m.k_init = "random"

    elseif m.k < 1
        warning *= "Number of clusters must be greater than 0. Defaulting to 3 clusters."
        m.k = 3

    elseif !(m.tol < 1.0)
        warning *= "Tolerance level must be less than 1. Defaulting to tol of 1e-6."
        m.tol = 1e-6

    elseif !(m.max_iters > 0)
        warning *= "Number of permitted iterations must be greater than 0. Defaulting to 300 iterations."
        m.max_iters = 300

    elseif !(m.threads > 0)
        warning *= "Number of threads must be at least 1. Defaulting to all threads available."
        m.threads = Threads.nthreads()

    elseif !(m.verbosity ∈ (0, 1))
        warning *= "Verbosity must be either 0 (no info) or 1 (info requested). Defaulting to 0."
        m.verbosity = 0
    end
    return warning
end


# TODO 3: implementation of fit, predict, and fitted_params of the model
####
#### FIT FUNCTION
####
"""
    TODO 3.1: Docs
    # fit the specified struct as a ParaKMeans model

    See also the [package documentation](https://pydatablog.github.io/ParallelKMeans.jl/stable).
"""
function MLJModelInterface.fit(m::KMeans, X)
    # convert tabular input data into the matrix model expects. Column assumed as features so input data is permuted
    if !m.copy
        # transpose input table without copying and pass to model
        DMatrix = convert(Array{Float64, 2}, X)'
    else
        # tranposes input table as a column major matrix after making a copy of the data
        DMatrix = MLJModelInterface.matrix(X; transpose=true)
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


"""
    TODO 3.2: Docs
"""
function MLJModelInterface.fitted_params(model::KMeans, fitresult)
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
"""
    TODO 3.3: Docs
"""
function MLJModelInterface.transform(m::KMeans, fitresult, Xnew)
    # make predictions/assignments using the learned centroids
    results = fitresult[1]
    DMatrix = MLJModelInterface.matrix(Xnew, transpose=true)

    # TODO 3.3.1: Warn users if fitresult is from a `non-converged` fit.
    # use centroid matrix to assign clusters for new data
    centroids = results.centers
    distances = Distances.pairwise(Distances.SqEuclidean(), DMatrix, centroids; dims=2)
    preds = argmin.(eachrow(distances))
    return MLJModelInterface.table(reshape(preds, :, 1), prototype=Xnew)
end


####
#### METADATA
####

# TODO 4: metadata for the package and for each of the model interfaces
metadata_pkg.(KMeans,
    name = "ParallelKMeans",
    uuid = "42b8e9d4-006b-409a-8472-7f34b3fb58af", # see your Project.toml
    url  = "https://github.com/PyDataBlog/ParallelKMeans.jl",  # URL to your package repo
    julia = true,          # is it written entirely in Julia?
    license = "MIT",       # your package license
    is_wrapper = false,    # does it wrap around some other package?
)


# Metadata for ParaKMeans model interface
metadata_model(KMeans,
    input   = MLJModelInterface.Table(MLJModelInterface.Continuous),
    output  = MLJModelInterface.Table(MLJModelInterface.Count),
    weights = false,
    descr   = ParallelKMeans_Desc,
	path	= "ParallelKMeans.KMeans")
