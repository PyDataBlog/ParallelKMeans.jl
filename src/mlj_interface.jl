# TODO 1: a using MLJModelInterface or import MLJModelInterface statement

####
#### MODEL DEFINITION
####
# TODO 2: MLJ-compatible model types and constructors
@mlj_model mutable struct KMeans <: MLJModelInterface.Unsupervised
    # Hyperparameters of the model
    algo::Symbol                            = :Lloyd::(_ in (:Lloyd, :Hamerly, :LightElkan))
    k_init::String                          = "k-means++"::(_ in ("k-means++", String)) # allow user seeding?
    k::Int                                  = 3::(_ > 0)
    tol::Float64                            = 1e-6::(_ < 1)
    max_iters::Int                          = 300::(_ > 0)
    copy::Bool                              = true
    threads::Int                            = Threads.nthreads()::(_ > 0)
    verbosity::Int                          = 0::(_ in (0, 1))  # Temp fix. Do we need to follow mlj verbosity style?
    init = nothing
end


# Expose all instances of user specified structs and package artifcats.
const ParallelKMeans_Desc = "Parallel & lightning fast implementation of all variants of the KMeans clustering algorithm in native Julia."

# availalbe variants for reference
const MLJDICT = Dict(:Lloyd => Lloyd(),
                     :Hamerly => Hamerly(),
                     :LightElkan => LightElkan())

# TODO 3: implementation of fit, predict, and fitted_params of the model
####
#### FIT FUNCTION
####
"""
    TODO 3.1: Docs

    See also the [package documentation](https://pydatablog.github.io/ParallelKMeans.jl/stable).
"""
function MLJModelInterface.fit(m::KMeans, X)
    # fit the specified struct as a ParaKMeans model

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
	path	= "ParallelKMeans.src.mlj_interface.KMeans")
