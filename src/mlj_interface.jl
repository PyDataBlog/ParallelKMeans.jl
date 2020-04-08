# TODO 1: a using MLJModelInterface or import MLJModelInterface statement
using MLJModelInterface
using ParallelKMeans


####
#### MODEL DEFINITION
####
# TODO 2: MLJ-compatible model types and constructors
@mlj_model mutable struct ParaKMeans <: MLJModelInterface.Unsupervised
    # Hyperparameters of the model
    algo::ParallelKMeans.AbstractKMeansAlg  = Lloyd()::(_ in (Lloyd(), Hamerly(), LightElkan()))
    k_init::String                          = "k-means++"::(_ in ("k-means++", String)) # allow user seeding?
    k::Int                                  = 3::(_ > 0)
    tol::Float64                            = 1e-6::(_ < 1)
    max_iters::Int                          = 300::(_ > 0)
    #transpose_type::String                  = "permute"::(_ in ("permute", "transpose"))
    threads::Int                            = Threads.nthreads()::(_ > 0)
    verbosity::Int                          = 0::(_ in (0, 1))
    init = nothing
end


# Expose all instances of user specified structs and package artifcats.
const KMeansModel = Union{ParaKMeans}
const ParallelKMeans_Desc = "Parallel & lightning fast implementation of all variants of the KMeans clustering algorithm in native Julia."


# TODO 3: implementation of fit, predict, and fitted_params of the model
####
#### FIT FUNCTION
####
"""
    TODO 3.1: Docs
"""
function MLJModelInterface.fit(m::ParaKMeans, verbosity::Int, X)
    # fit the specified struct as a ParaKMeans model

    # assumes user supplied table with columns as features
    DMatrix = MLJModelInterface.matrix(X; transpose=true)
    
    # fit model and get results
    if m.verbosity > 0
		fitresult = ParallelKMeans.kmeans(m.algo, DMatrix, m.k;
	                                      n_threads = m.threads, k_init=m.k_init,
	                                      max_iters=m.max_iters, tol=m.tol, init=m.init,
										  verbose=true)
    else
		fitresult = ParallelKMeans.kmeans(m.algo, DMatrix, m.k;
	                                      n_threads = m.threads, k_init=m.k_init,
	                                      max_iters=m.max_iters, tol=m.tol, init=m.init,
										  verbose=false)
    end

    cache = nothing
    report = NamedTuple{}()

    return (fitresult, cache, report)
end


"""
    TODO 3.2: Docs
"""
function MLJModelInterface.fitted_params(model::KMeansModel, fitresult)
    # extract what's relevant from `fitresult`
    centres = fitresult.centres
    converged = fitresult.converged
    iters = fitresult.iterations
    totalcost = fitresult.totalcost
    # then return as a NamedTuple
    return (centres = centres, totalcost = totalcost, iterations = iters, converged = converged)
end


####
#### PREDICT FUNCTION
####
"""
    TODO 3.3: Docs
"""
function MLJModelInterface.predict(m::ParaKMeans, fitresult, Xnew)
    # ...
end


####
#### METADATA
####

# TODO 4: metadata for the package and for each of the model interfaces
metadata_pkg.(KMeansModel,
    name = "ParallelKMeans",
    uuid = "42b8e9d4-006b-409a-8472-7f34b3fb58af", # see your Project.toml
    url  = "https://github.com/PyDataBlog/ParallelKMeans.jl",  # URL to your package repo
    julia = true,          # is it written entirely in Julia?
    license = "MIT",       # your package license
    is_wrapper = false,    # does it wrap around some other package?
)


# Metadata for ParaKMeans model interface
metadata_model(ParaKMeans,
    input   = MLJModelInterface.Table(MLJModelInterface.Continuous),  # what input data is supported?           # for a supervised model, what target?
    output  = MLJModelInterface.Table(MLJModelInterface.Count),  # for an unsupervised, what output?
    weights = false,                                             # does the model support sample weights?
    descr   = ParallelKMeans_Desc,
	path	= "ParallelKMeans.src.mlj_interface.ParaKMeans"
	#path    = "YourPackage.SubModuleContainingModelStructDefinition.YourModel1"
    )
