# ParallelKMeans

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://PyDataBlog.github.io/ParallelKMeans.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://PyDataBlog.github.io/ParallelKMeans.jl/dev)
[![Build Status](https://www.travis-ci.org/PyDataBlog/ParallelKMeans.jl.svg?branch=master)](https://www.travis-ci.org/PyDataBlog/ParallelKMeans.jl)
[![Coverage Status](https://coveralls.io/repos/github/PyDataBlog/ParallelKMeans.jl/badge.svg?branch=master)](https://coveralls.io/github/PyDataBlog/ParallelKMeans.jl?branch=master)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FPyDataBlog%2FParallelKMeans.jl.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FPyDataBlog%2FParallelKMeans.jl?ref=badge_shield)
_________________________________________________________________________________________________________
**Authors:** [Bernard Brenyah](https://www.linkedin.com/in/bbrenyah/) & [Andrey Oskin](https://www.linkedin.com/in/andrej-oskin-b2b03959/)
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
pkg> dev git@github.com:PyDataBlog/ParallelKMeans.jl.git
```

Don't forget to checkout the experimental branch and you are good to go with bleeding edge features and breaks!
```bash
git checkout experimental
```
_________________________________________________________________________________________________________

### Features

- Lightening fast implementation of Kmeans clustering algorithm even on a single thread in native Julia.
- Support for multi-theading implementation of Kmeans clustering algorithm.
- Kmeans++ initialization for faster and better convergence.
- Modified version of Elkan's Triangle inequality to speed up K-Means algorithm.

_________________________________________________________________________________________________________

### License

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FPyDataBlog%2FParallelKMeans.jl.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2FPyDataBlog%2FParallelKMeans.jl?ref=badge_large)
