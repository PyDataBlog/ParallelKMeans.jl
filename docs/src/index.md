# ParallelKMeans.jl Package

```@contents
Depth = 4
```

## Motivation
It's actually a funny story led to the development of this package.
What started off as a personal toy project trying to re-construct the K-Means algorithm in native Julia blew up after a heated discussion on the Julia Discourse forum when I asked for Julia optimizaition tips. Long story short, Julia community is an amazing one! Andrey offered his help and together, we decided to push the speed limits of Julia with a parallel implementation of the most famous clustering algorithm. The initial results were mind blowing so we have decided to tidy up the implementation and share with the world as a maintained Julia pacakge. 

Say hello to `ParallelKMeans`!

This package aims to utilize the speed of Julia and parallelization (both CPU & GPU) to offer an extremely fast implementation of the K-Means clustering algorithm and its variations via a friendly interface for practioners.

In short, we hope this package will eventually mature as the "one stop" shop for everything KMeans on both CPUs and GPUs.

## K-Means Algorithm Implementation Notes
Since Julia is a column major language, the input (design matrix) expected by the package in the following format;

- Design matrix X of size n×m, the i-th column of X `(X[:, i])` is a single data point in n-dimensional space.
- Thus, the rows of the design design matrix represents the feature space with the columns representing all the training examples in this feature space.

One of the pitfalls of K-Means algorithm is that it can fall into a local minima. 
This implementation inherits this problem like every implementation does.
As a result, it is useful in practice to restart it several times to get the correct results.

## Installation
You can grab the latest stable version of this package from Julia registries by simply running;

*NB:* Don't forget to Julia's package manager with `]`

```julia
pkg> add ParallelKMeans
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
- 'Kmeans++' initialization for faster and better convergence.
- Modified version of Elkan's Triangle inequality to speed up K-Means algorithm.


## Pending Features
- [X] Implementation of [Hamerly implementation](https://www.researchgate.net/publication/220906984_Making_k-means_Even_Faster). 
- [ ] Full Implementation of Triangle inequality based on [Elkan - 2003 Using the Triangle Inequality to Accelerate K-Means"](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf).
- [ ] Implementation of [Geometric methods to accelerate k-means algorithm](http://cs.baylor.edu/~hamerly/papers/sdm2016_rysavy_hamerly.pdf).
- [ ] Support for DataFrame inputs.
- [ ] Refactoring and finalizaiton of API desgin.
- [ ] GPU support.
- [ ] Even faster Kmeans implementation based on current literature.
- [ ] Optimization of code base.
- [ ] Improved Documentation
- [ ] More benchmark tests


## How To Use
Taking advantage of Julia's brilliant multiple dispatch system, the package exposes users to a very easy to use API. 

```julia
using ParallelKMeans

# Uses all available CPU cores by default
multi_results = kmeans(X, 3; max_iters=300)

# Use only 1 core of CPU
results = kmeans(X, 3; n_threads=1, max_iters=300)
```

The main design goal is to offer all available variations of the KMeans algorithm to end users as composable elements. By default, Lloyd's implementation is used but users can specify different variations of the KMeans clustering algorithm via this interface

```julia
some_results = kmeans([algo], input_matrix, k; kwargs)

# example
r = kmeans(Lloyd(), X, 3)  # same result as the default 
```

### Supported KMeans algorithm variations.
- [Lloyd()](https://cs.nyu.edu/~roweis/csc2515-2006/readings/lloyd57.pdf) 
- [Hamerly()](https://www.researchgate.net/publication/220906984_Making_k-means_Even_Faster) 
- [Geometric()](http://cs.baylor.edu/~hamerly/papers/sdm2016_rysavy_hamerly.pdf) - (Coming soon)
- [Elkan()](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf) - (Coming soon) 
- [MiniBatch()](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf) - (Coming soon)


### Practical Usage Examples
Some of the common usage examples of this package are as follows:

#### Clustering With A Desired Number Of Groups

```julia 
using ParallelKMeans, RDatasets, Plots

# load the data
iris = dataset("datasets", "iris"); 

# features to use for clustering
features = collect(Matrix(iris[:, 1:4])'); 

# various artificats can be accessed from the result ie assigned labels, cost value etc
result = kmeans(features, 3); 

# plot with the point color mapped to the assigned cluster index
scatter(iris.PetalLength, iris.PetalWidth, marker_z=result.assignments,
        color=:lightrainbow, legend=false)

```

![Image description](iris_example.jpg)

#### Elbow Method For The Selection Of optimal number of clusters
```julia
using ParallelKMeans

# Single Thread Implementation of Lloyd's Algorithm
b = [ParallelKMeans.kmeans(X, i, n_threads=1;
                           tol=1e-6, max_iters=300, verbose=false).totalcost for i = 2:10]

# Multi Thread Implementation of Lloyd's Algorithm by default
c = [ParallelKMeans.kmeans(X, i; tol=1e-6, max_iters=300, verbose=false).totalcost for i = 2:10]

```


## Benchmarks
Currently, this package is benchmarked against similar implementation in both Python and Julia. All reproducible benchmarks can be found in [ParallelKMeans/extras](https://github.com/PyDataBlog/ParallelKMeans.jl/tree/master/extras) directory. More tests in various languages are planned beyond the initial release version (`0.1.0`).

*Note*: All benchmark tests are made on the same computer to help eliminate any bias. 


Currently, the benchmark speed tests are based on the search for optimal number of clusters using the [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)) since this is a practical use case for most practioners employing the K-Means algorithm. 


![benchmark_image.png](benchmark_image.png)


## Release History 
- 0.1.0 Initial release


## Contributing
Ultimately, we see this package as potentially the one stop shop for everything related to KMeans algorithm and its speed up variants. We are open to new implementations and ideas from anyone interested in this project.

Detailed contribution guidelines will be added in upcoming releases.

<!--- Insert Contribution Guidelines Below --->

```@index
```

```@autodocs
Modules = [ParallelKMeans]
```
