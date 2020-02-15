# Local scripts usage

This mode is useful when multiple changes are being made and one do not want to generate multiple commits per branch.

## Single benchmark
To run benchmarks locally

```julia
julia runbenchmarks.jl
```

To see the results of the last benchmark
```julia
julia pprintresult.jl | less
```

To see the results of previous run
```julia
julia pprintresult.jl results/20200101T010101.json | less
```

## Judge
One may compare results of two runs with the following command

```julia
julia pprintjudge.jl results/20200102T000000.json results/20200101T0000000.json
```
here first argument is target, second is baseline

Without any arguments `pprintjudge` generates comparison of two the last two runs or
comparison of last run with the last baseline run if it exists. To generate baseline run
use the following

```julia
julia runbenchmarks.jl -b   # creates file of the form results/20200101T000000-baseline.json
```

All other runs will be compared to this file.

# BenchmarkCI

This mode is useful to compare different branches or for automated benchmarking.

Detailed information regarding running BenchmarkCI can be found in [BenchmarkCI](https://github.com/tkf/BenchmarkCI.jl) documentation.

Following commands will generate benchmark report that compares current commit with "origin/master"
```julia
shell> cd ~/.julia/dev/MyProject/

julia> using BenchmarkCI

julia> BenchmarkCI.judge()
...

julia> BenchmarkCI.displayjudgement()
```

If one need to run benchmark against local master, than instead of `BenchmarkCI.judge()` one should
use `BenchmarkCI.judge(baseline="master")`
