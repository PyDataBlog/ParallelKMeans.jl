module ParallelKMeans

using StatsBase
using Random
using UnsafeArrays
import MLJModelInterface
import Base.Threads: @spawn
import Distances
import Distances: Euclidean, evaluate

include("kmeans.jl")
include("seeding.jl")
include("lloyd.jl")
include("hamerly.jl")
include("elkan.jl")
include("yinyang.jl")
include("coreset.jl")
include("mlj_interface.jl")

export kmeans
export Lloyd, Hamerly, Elkan, Yinyang, 阴阳, Coreset

end # module
