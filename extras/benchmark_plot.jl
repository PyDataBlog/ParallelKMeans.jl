using Plots
using CSV
using DataFrames
using Chain


data = CSV.read("extras/updated_benchmarks_may_1.csv", DataFrame)

long_data = @chain data begin
    rename(_, [replace(x, " " => "_") for x in names(_)])
    rename(_, [replace(x, "_sample_(secs)" => "") for x in names(_)])
    filter(:process => !=("stochastic"), _)
    stack(_, 1:4)
end

color_map = Dict{String, Int}()

for (idx, val) in enumerate(unique(long_data.package))
    push!(color_map, val => idx)
end

assign_linestyle(x) = occursin("ParallelKMeans", x) ? :solid : :dashdot

function assign_rank(x)
    if x == "1_million"
        return 4
    elseif x == "100k"
        return 3
    elseif x == "10k"
        return 2
    elseif x == "1k"
        return 1
    end
end

long_data[:, "linestyle"] = assign_linestyle.(long_data.package);
long_data[:, "rank"] = assign_rank.(long_data.variable);


plt = plot(title = "Elbow Method Benchmark Results",
           yaxis=:log,
           palette=:seaborn_deep,
           size=(1000, 700),
           ylabel="Execution Time (in seconds - logged)",
           yrotation=30,
           xlabel="Sample Sizes",
           legend=:topleft)

for pkg in unique(long_data.package)
    pkg_data = filter(:package => ==(pkg), long_data)
    sort!(pkg_data, order(:rank, rev=false))
    plot!(pkg_data.variable,
         pkg_data.value,
         lw=3,
         linestyle=pkg_data.linestyle,
         label=pkg,
         color=color_map[pkg])
    scatter!(pkg_data.variable, pkg_data.value, markersize=4, color=color_map[pkg], label="")
end

display(plt)
