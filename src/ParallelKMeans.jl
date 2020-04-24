module ParallelKMeans

using StatsBase
import MLJModelInterface
import Base.Threads: @spawn
import Distances

const MMI = MLJModelInterface

include("seeding.jl")
include("kmeans.jl")
include("lloyd.jl")
include("hamerly.jl")
include("elkan.jl")
include("yinyang.jl")
include("mlj_interface.jl")

export kmeans
export Lloyd, Hamerly, Elkan, Yinyang

end # module
