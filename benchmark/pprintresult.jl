using PkgBenchmark
include("pprinthelper.jl")

if length(ARGS) == 1
    result = PkgBenchmark.readresults(ARGS[1])
elseif length(ARGS) == 0
    res_dir = joinpath(@__DIR__, "results")
    path = maximum(readdir(res_dir))
    path = joinpath(res_dir, path)
    result = PkgBenchmark.readresults(path)
end

displayresult(result)
