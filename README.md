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

1. [Motivation](#Motivatiion)
2. [Installation](#Installation)
3. [Features](#Features)
4. [Benchmarks](#Benchmarks)
5. [Pending Features](#Pending-Features)
6. [How To Use](#How-To-Use)
7. [Release History](#Release-History)
8. [How To Contribute](#How-To-Contribute)
9. [Credits](#Credits)
10. [License](#License)

_________________________________________________________________________________________________________

### Motivation
It's a funny story actually led to the development of this package.
What started off as a personal toy project trying to re-construct the K-Means algorithm in  native Julia blew up after into a heated discussion on the Julia Discourse forums after I asked for Julia optimizaition tips. Long story short, Julia community is an amazing one! Andrey Oskin offered his help and together, we decided to push the speed limits of Julia with a parallel implementation of the most famous clustering algorithm. The initial results were mind blowing so we have decided to tidy up the implementation and share with the world. 

Say hello to our baby, `ParallelKMeans`!
_________________________________________________________________________________________________________

### Installation
You can grab the latest stable version of this package by simply running in Julia.
Don't forget to Julia's package manager with `]`

```julia
pkg> add TextAnalysis
```

For the few (and selected) brave ones, one can simply grab the current experimental features by simply adding the experimental branch to your development environment after invoking the package manager with `]`:

```julia
dev git@github.com:PyDataBlog/ParallelKMeans.jl.git
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

### Benchmarks

_________________________________________________________________________________________________________

### Pending Features
- [X] Implementation of Triangle inequality based on [Elkan C. (2003) "Using the Triangle Inequality to Accelerate
K-Means"](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf)
- [ ] Support for DataFrame inputs.
- [ ] Refactoring and finalizaiton of API desgin.
- [ ] GPU support.
- [ ] Even faster Kmeans implementation based on current literature.
- [ ] Optimization of code base.

_________________________________________________________________________________________________________

### How To Use

```Julia

```

_________________________________________________________________________________________________________

### Release History

- 0.1.0 Initial release

_________________________________________________________________________________________________________

### How To Contribue

_________________________________________________________________________________________________________

### Credits

_________________________________________________________________________________________________________

### License

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FPyDataBlog%2FParallelKMeans.jl.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2FPyDataBlog%2FParallelKMeans.jl?ref=badge_large)
