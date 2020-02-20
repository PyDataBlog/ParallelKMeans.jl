using Documenter, ParallelKMeans

makedocs(;
    modules=[ParallelKMeans],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/PyDataBlog/ParallelKMeans.jl/blob/{commit}{path}#L{line}",
    sitename="ParallelKMeans.jl",
    authors="Bernard Brenyah & Andrey Oskin",
    assets=String[],
)

deploydocs(;
    repo="github.com/PyDataBlog/ParallelKMeans.jl",
)
