using Documenter, ParallelKMeans

makedocs(;
    modules=[ParallelKMeans],
    authors="Bernard Brenyah & Andrey Oskin",
    repo="https://github.com/PyDataBlog/ParallelKMeans.jl/blob/{commit}{path}#L{line}",
    sitename="ParallelKMeans.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://PyDataBlog.github.io/ParallelKMeans.jl",
        siteurl="https://github.com/PyDataBlog/ParallelKMeans.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/PyDataBlog/ParallelKMeans.jl",
)
