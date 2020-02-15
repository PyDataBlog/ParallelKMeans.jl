using PkgBenchmark
include("pprinthelper.jl")

if length(ARGS) == 2
    group_target = PkgBenchmark.readresults(ARGS[1])
    group_baseline = PkgBenchmark.readresults(ARGS[2])
else
    res_dir = joinpath(@__DIR__, "results")
    last_result = maximum(filter(x -> x != ".gitignore", readdir(res_dir)))
    baselines = filter(x -> (x != ".gitignore") & occursin("baseline", x), readdir(res_dir))
    last_baseline = isempty(baselines) ? maximum(filter(x -> (x != ".gitignore") & (x != last_result), readdir(res_dir))) : maximum(baselines)
    group_target = PkgBenchmark.readresults(joinpath(res_dir, last_result))
    group_baseline = PkgBenchmark.readresults(joinpath(res_dir, last_baseline))
end
judgement = judge(group_target, group_baseline)

displayresult(judgement)

printnewsection("Target result")
displayresult(group_target)

printnewsection("Baseline result")
displayresult(group_baseline)
