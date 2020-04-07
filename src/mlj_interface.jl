# TODO 1: a using MLJModelInterface or import MLJModelInterface statement
using MLJModelInterface
using ParallelKMeans


####
#### MODEL DEFINITION
####
# TODO 2: MLJ-compatible model types and constructors,
@mlj_model mutable struct ParaKMeans <: MLJModelInterface.Unsupervised
    # Hyperparameters of the model
    algo::AbstractKMeansAlg     = Lloyd::(_ in (Lloyd, Hamerly, Elkan))
    k_init::String              = "k-means++"::(_ in ("k-means++", String))
    k::Int                      = 3::(_ > 0)
    tol::Float                  = 1e-6::(_ < 1)
    max_iters::Int              = 300::(_ > 0)
end


# TODO 3: implementation of fit, predict, and fitted_params of the model
####
#### FIT FUNCTION
####

function MLJModelInterface.fit(m::ParaKMeans, verbosity::Int, X, y, w=nothing)
    # body ...
    return (fitresult, cache, report)
end


function MLJModelInterface.fitted_params(model::ParaKMeans, fitresult)
    # extract what's relevant from `fitresult`
    # ...
    # then return as a NamedTuple
    return (learned_param1 = ..., learned_param2 = ...)
end


####
#### PREDICT FUNCTION
####
function MLJModelInterface.predict(m::ParaKMeans, fitresult, Xnew)
    # ...
end


####
#### METADATA
####

# TODO 4: metadata for the package and for each of your models
const PARAKMEANS_MODELS = Union{ParaKMeans}

metadata_pkg.(PARAKMEANS_MODELS,
    name = "ParallelKMeans",
    uuid = "42b8e9d4-006b-409a-8472-7f34b3fb58af", # see your Project.toml
    url  = "https://github.com/PyDataBlog/ParallelKMeans.jl",  # URL to your package repo
    julia = true,          # is it written entirely in Julia?
    license = "MIT",       # your package license
    is_wrapper = false,    # does it wrap around some other package?
)


# Metadata for ParaKMeans model
metadata_model(ParaKMeans,
    input   = MLJModelInterface.Table(MLJModelInterface.Continuous),  # what input data is supported?           # for a supervised model, what target?
    output  = MLJModelInterface.Table(MLJModelInterface.Count),  # for an unsupervised, what output?
    weights = false,                                             # does the model support sample weights?
    descr   = "Parallel & lightning fast implementation of all variants of the KMeans clustering algorithm in native Julia.",
	path	= "ParallelKMeans.src.mlj_interface.ParaKMeans"
	#path    = "YourPackage.SubModuleContainingModelStructDefinition.YourModel1"
    )
