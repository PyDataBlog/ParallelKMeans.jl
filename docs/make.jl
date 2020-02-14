using Documenter, ParallelKMeans

makedocs(;
    modules=[ParallelKMeans],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/Arkoniak/ParallelKMeans.jl/blob/{commit}{path}#L{line}",
    sitename="ParallelKMeans.jl",
    authors="Andrey Oskin",
    assets=String[],
)

deploydocs(;
    repo="github.com/Arkoniak/ParallelKMeans.jl",
)
