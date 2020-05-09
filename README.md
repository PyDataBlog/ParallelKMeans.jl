# ParallelKMeans

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://PyDataBlog.github.io/ParallelKMeans.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://PyDataBlog.github.io/ParallelKMeans.jl/dev)
[![Build Status](https://www.travis-ci.org/PyDataBlog/ParallelKMeans.jl.svg?branch=master)](https://www.travis-ci.org/PyDataBlog/ParallelKMeans.jl)
[![Coverage Status](https://coveralls.io/repos/github/PyDataBlog/ParallelKMeans.jl/badge.svg?branch=master)](https://coveralls.io/github/PyDataBlog/ParallelKMeans.jl?branch=master)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FPyDataBlog%2FParallelKMeans.jl.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FPyDataBlog%2FParallelKMeans.jl?ref=badge_shield)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PyDataBlog/ParallelKMeans.jl/master)
_________________________________________________________________________________________________________
**Authors:** [Bernard Brenyah](https://www.linkedin.com/in/bbrenyah/) & [Andrey Oskin](https://www.linkedin.com/in/andrej-oskin-b2b03959/)
_________________________________________________________________________________________________________

<div align="center">
    <b>Classic & Contemporary Variants Of K-Means In Sonic Mode</b>
</div>

<p align="center">
  <img src="https://user-images.githubusercontent.com/2630519/80216880-70b60b00-8647-11ea-913b-7977ef1c156c.gif">
</p>

_________________________________________________________________________________________________________

## Table Of Content

1. [Documentation](#Documentation)
2. [Installation](#Installation)
3. [Features](#Features)
4. [License](#License)

_________________________________________________________________________________________________________

### Documentation

- Stable Documentation: [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://PyDataBlog.github.io/ParallelKMeans.jl/stable)

- Experimental Documentation: [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://PyDataBlog.github.io/ParallelKMeans.jl/dev)

_________________________________________________________________________________________________________

### Installation

You can grab the latest stable version of this package by simply running in Julia.
Don't forget to Julia's package manager with `]`

```julia
pkg> add ParallelKMeans
```

For the few (and selected) brave ones, one can simply grab the current experimental features by simply adding the experimental branch to your development environment after invoking the package manager with `]`:

```julia
pkg> add ParallelKMeans#experimental
```

To revert to a stable version, you can simply run:

```julia
pkg> free ParallelKMeans
```

_________________________________________________________________________________________________________

### Features

- Lightening fast implementation of K-Means clustering algorithm even on a single thread in native Julia.
- Support for multi-theading implementation of K-Means clustering algorithm.
- Kmeans++ initialization for faster and better convergence.
- Implementation of all available variants of the K-Means algorithm.
- Support for all distance metrics available at [Distances.jl](https://github.com/JuliaStats/Distances.jl)
- Supported interface as an [MLJ](https://github.com/alan-turing-institute/MLJ.jl#available-models) model.

_________________________________________________________________________________________________________

### Benchmarks

Currently, this package is benchmarked against similar implementations in both Python, R, and Julia. All reproducible benchmarks can be found in [ParallelKMeans/extras](https://github.com/PyDataBlog/ParallelKMeans.jl/tree/master/extras) directory.

![benchmark_image.png](docs/src/benchmark_image.png)
_________________________________________________________________________________________________________

### License

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FPyDataBlog%2FParallelKMeans.jl.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2FPyDataBlog%2FParallelKMeans.jl?ref=badge_large)
