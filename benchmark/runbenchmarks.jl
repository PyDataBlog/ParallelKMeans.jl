using PkgBenchmark
using Dates

if (length(ARGS) == 1) && (ARGS[1] == "-b")
    path =  joinpath(@__DIR__, "results", "$(Dates.format(now(), dateformat"yyyymmddTHHMMSS"))-baseline.json")
else
    path =  joinpath(@__DIR__, "results", "$(Dates.format(now(), dateformat"yyyymmddTHHMMSS")).json")
end

benchmarkpkg(
    dirname(@__DIR__),
    BenchmarkConfig(
        env = Dict(
            "JULIA_NUM_THREADS" => "1",
            "OMP_NUM_THREADS" => "1",
        ),
    ),
    resultfile = path,
)
