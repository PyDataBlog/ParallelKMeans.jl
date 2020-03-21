# ParallelKMeans.jl Package

```@contents
Depth = 4
```

## Motivation
It's actually a funny story led to the development of this package.
What started off as a personal toy project trying to re-construct the K-Means algorithm in  native Julia blew up after into a heated discussion on the Julia Discourse forums after I asked for Julia optimizaition tips. Long story short, Julia community is an amazing one! Andrey Oskin offered his help and together, we decided to push the speed limits of Julia with a parallel implementation of the most famous clustering algorithm. The initial results were mind blowing so we have decided to tidy up the implementation and share with the world as a maintained Julia pacakge. 

Say hello to our baby, `ParallelKMeans`!

This package aims to utilize the speed of Julia and parallelization (both CPU & GPU) by offering an extremely fast implementation of the K-Means clustering algorithm with user friendly interface.


## K-Means Algorithm Implementation Notes
Explain main algos and some few lines about the input dimension as well as 

## Installation
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

## Features
- Lightening fast implementation of Kmeans clustering algorithm even on a single thread in native Julia.
- Support for multi-theading implementation of Kmeans clustering algorithm.
- Kmeans++ initialization for faster and better convergence.
- Modified version of Elkan's Triangle inequality to speed up K-Means algorithm.


## Pending Features
- [X] Implementation of Triangle inequality based on [Elkan C. (2003) "Using the Triangle Inequality to Accelerate
K-Means"](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf)
- [ ] Support for DataFrame inputs.
- [ ] Refactoring and finalizaiton of API desgin.
- [ ] GPU support.
- [ ] Even faster Kmeans implementation based on current literature.
- [ ] Optimization of code base.


## How To Use
Taking advantage of Julia's brilliant multiple dispatch system, the package exposes users to a very easy to use API.

```julia
using ParallelKMeans

# Use only 1 core of CPU
results = kmeans(X, 3, ParallelKMeans.SingleThread(), tol=1e-6, max_iters=300)

# Use all available CPU cores
multi_results = kmeans(X, 3, ParallelKMeans.MultiThread(), tol=1e-6, max_iters=300)
```

### Practical Usage Examples
Some of the common usage examples of this package are as follows:

#### Clustering With A Desired Number Of Groups

```julia 
using ParallelKMeans, RDatasets, Plots

# load the data
iris = dataset("datasets", "iris"); 

# features to use for clustering
features = collect(Matrix(iris[:, 1:4])'); 

result = kmeans(features, 3, ParallelKMeans.MultiThread()); 

# plot with the point color mapped to the assigned cluster index
scatter(iris.PetalLength, iris.PetalWidth, marker_z=result.assignments,
        color=:lightrainbow, legend=false)

# TODO: Add scatter plot image
```

#### Elbow Method For The Selection Of optimal number of clusters
```julia
using ParallelKMeans

# Single Thread Implementation of Lloyd's Algorithm
b = [ParallelKMeans.kmeans(X, i, ParallelKMeans.SingleThread(),
                          tol=1e-6, max_iters=300, verbose=false).totalcost for i = 2:10]

# Multi Thread Implementation of Lloyd's Algorithm
c = [ParallelKMeans.kmeans(X, i, ParallelKMeans.MultiThread(), 
                           tol=1e-6, max_iters=300, verbose=false).totalcost for i = 2:10]

# Multi Thread Implementation plus a modified version of Elkan's triangiulity of inequaltiy
# to boost speed
d = [ParallelKMeans.kmeans(ParallelKMeans.LightElkan(), X, i, ParallelKMeans.MultiThread(),
                           tol=1e-6, max_iters=300, verbose=false).totalcost for i = 2:10]

# Single Thread Implementation plus a modified version of Elkan's triangiulity of inequaltiy
# to boost speed
e = [ParallelKMeans.kmeans(ParallelKMeans.LightElkan(), X, i, ParallelKMeans.SingleThread(),
                           tol=1e-6, max_iters=300, verbose=false).totalcost for i = 2:10]
```


## Benchmarks


## Release History 
- 0.1.0 Initial release


## Contributing


```@index
```

```@autodocs
Modules = [ParallelKMeans]
```
