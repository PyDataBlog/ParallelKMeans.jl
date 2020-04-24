# [ParallelKMeans.jl Package](https://github.com/PyDataBlog/ParallelKMeans.jl)

## Motivation

It's actually a funny story led to the development of this package.
What started off as a personal toy project trying to re-construct the K-Means algorithm in native Julia blew up after a heated discussion on the Julia Discourse forum when I asked for Julia optimization tips. Long story short, Julia community is an amazing one! Andrey offered his help and together, we decided to push the speed limits of Julia with a parallel implementation of the most famous clustering algorithm. The initial results were mind blowing so we have decided to tidy up the implementation and share with the world as a maintained Julia pacakge.

Say hello to `ParallelKMeans`!

This package aims to utilize the speed of Julia and parallelization (both CPU & GPU) to offer an extremely fast implementation of the K-Means clustering algorithm and its variants via a friendly interface for practioners.

In short, we hope this package will eventually mature as the "one-stop-shop" for everything K-Means on CPUs and GPUs.

## K-Means Algorithm Implementation Notes

Since Julia is a column major language, the input (design matrix) expected by the package must be in the following format;

- Design matrix X of size n×m, the i-th column of X `(X[:, i])` is a single data point in n-dimensional space.
- Thus, the rows of the design matrix represent the feature space with the columns representing all the training samples in this feature space.

One of the pitfalls of K-Means algorithm is that it can fall into a local minima.
This implementation inherits this problem like every implementation does.
As a result, it is useful in practice to restart it several times to get the correct results.

## Installation

If you are using  Julia in the recommended [Juno IDE](https://junolab.org/), the number of threads is already set to the number of available CPU cores so multithreading enabled out of the box.
For other IDEs, multithreading must be exported in your environment before launching the Julia REPL in the command line.

*TIP*: One needs to navigate or point to the Julia executable file to be able to launch it in the command line.
Enable multi threading on Mac/Linux systems via;

```bash
export JULIA_NUM_THREADS=n  # where n is the number of threads/cores
```

For Windows systems:

```bash
set JULIA_NUM_THREADS=n  # where n is the number of threads/cores
```

You can grab the latest stable version of this package from Julia registries by simply running;

*NB:* Don't forget to invoke Julia's package manager with `]`

```julia
pkg> add ParallelKMeans
```

The few (and selected) brave ones can simply grab the current experimental features by simply adding the experimental branch to your development environment after invoking the package manager with `]`:

```julia
dev git@github.com:PyDataBlog/ParallelKMeans.jl.git
```

Don't forget to checkout the experimental branch and you are good to go with bleeding edge features and breakages!

```bash
git checkout experimental
```

## Features

- Lightning fast implementation of Kmeans clustering algorithm even on a single thread in native Julia.
- Support for multi-threading implementation of K-Means clustering algorithm.
- 'Kmeans++' initialization for faster and better convergence.
- Implementation of available classic and contemporary variants of the K-Means algorithm.

## Pending Features

- [X] Implementation of [Hamerly implementation](https://www.researchgate.net/publication/220906984_Making_k-means_Even_Faster).
- [X] Interface for inclusion in Alan Turing Institute's [MLJModels](https://github.com/alan-turing-institute/MLJModels.jl#who-is-this-repo-for).
- [X] Full Implementation of Triangle inequality based on [Elkan - 2003 Using the Triangle Inequality to Accelerate K-Means"](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf).
- [X] Implementation of [Yinyang K-Means: A Drop-In Replacement of the Classic K-Means with Consistent Speedup](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ding15.pdf)
- [ ] Implementation of [Geometric methods to accelerate k-means algorithm](http://cs.baylor.edu/~hamerly/papers/sdm2016_rysavy_hamerly.pdf).
- [ ] Support for other distance metrics supported by [Distances.jl](https://github.com/JuliaStats/Distances.jl#supported-distances).
- [ ] Support of MLJ Random generation hyperparameter.
- [ ] Native support for tabular data inputs outside of MLJModels' interface.
- [ ] Refactoring and finalizaiton of API desgin.
- [ ] GPU support.
- [ ] Distributed calculations support.
- [ ] Implementation of other K-Means algorithm variants based on recent literature.
- [ ] Optimization of code base.
- [ ] Improved Documentation
- [ ] More benchmark tests.

## How To Use

Taking advantage of Julia's brilliant multiple dispatch system, the package exposes users to a very easy-to-use API.

```julia
using ParallelKMeans

# Uses all available CPU cores by default
multi_results = kmeans(X, 3; max_iters=300)

# Use only 1 core of CPU
results = kmeans(X, 3; n_threads=1, max_iters=300)
```

The main design goal is to offer all available variations of the KMeans algorithm to end users as composable elements. By default, Lloyd's implementation is used but users can specify different variations of the KMeans clustering algorithm via this interface;

```julia
some_results = kmeans([algo], input_matrix, k; kwargs)

# example
r = kmeans(Lloyd(), X, 3)  # same result as the default
```

```julia
# r contains all the learned artifacts that can be accessed as;
r.centers               # cluster centers (d x k)
r.assignments           # label assignments (n)
r.totalcost             # total cost (i.e. objective)
r.iterations            # number of elapsed iterations
r.converged             # whether the procedure converged
```

### Supported KMeans algorithm variations and recommended use cases

- [Lloyd()](https://cs.nyu.edu/~roweis/csc2515-2006/readings/lloyd57.pdf)  - Default algorithm but only recommended for very small matrices (switch to `n_threads = 1` to avoid overhead).
- [Hamerly()](https://www.researchgate.net/publication/220906984_Making_k-means_Even_Faster) - Hamerly is good for moderate number of clusters (< 50?) and moderate dimensions (<100?).
- [Elkan()](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf) - Recommended for high dimensional data.
- [Yinyang()](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ding15.pdf) - Recommended for large dimensions and/or large number of clusters.
- [Geometric()](http://cs.baylor.edu/~hamerly/papers/sdm2016_rysavy_hamerly.pdf) - (Coming soon)
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

# various artifacts can be accessed from the result i.e. assigned labels, cost value etc
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
b = [ParallelKMeans.kmeans(X, i, n_threads=1; tol=1e-6, max_iters=300, verbose=false).totalcost for i = 2:10]

# Multi-threaded Implementation of Lloyd's Algorithm by default
c = [ParallelKMeans.kmeans(X, i; tol=1e-6, max_iters=300, verbose=false).totalcost for i = 2:10]

```

## Benchmarks

Currently, this package is benchmarked against similar implementations in both Python and Julia. All reproducible benchmarks can be found in [ParallelKMeans/extras](https://github.com/PyDataBlog/ParallelKMeans.jl/tree/master/extras) directory. More tests in various languages are planned beyond the initial release version (`0.1.0`).

*Note*: All benchmark tests are made on the same computer to help eliminate any bias.
|PC Name                      |CPU                       |Ram               |
|:---------------------------:|:------------------------:|:----------------:|
|iMac (Retina 5K 27-inch 2019)|3 GHz 6-Core Intel Core i5|8 GB 2667 MHz DDR4|

Currently, the benchmark speed tests are based on the search for optimal number of clusters using the [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)) since this is a practical use case for most practioners employing the K-Means algorithm.

### Benchmark Results

![benchmark_image.png](benchmark_image.png)

_________________________________________________________________________________________________________

|1 million sample (secs)|100k sample (secs)|10k sample (secs)|1k sample (secs)|package                |language   |
|:---------------------:|:----------------:|:---------------:|:--------------:|:---------------------:|:---------:|
|       538.53100       |     33.15700     |     0.74238     |    0.01710     |     Clustering.jl     |   Julia   |
|       220.35700       |     20.93600     |     0.82430     |    0.02639     |        mlpack         |C++ Wrapper|
|       20.55400        |     2.91300      |     0.17559     |    0.00609     |         Lloyd         |   Julia   |
|       11.51800        |     0.96637      |     0.09990     |    0.00635     |        Hamerly        |   Julia   |
|       14.01900        |     1.13100      |     0.07912     |    0.00646     |         Elkan         |   Julia   |
|        9.97000        |     1.14600      |     0.10834     |    0.00704     |       YingYang        |   Julia   |
|      1,430.00000      |    146.00000     |     5.77000     |    0.34400     |    Sklearn Kmeans     |  Python   |
|       30.10000        |     3.75000      |     0.61300     |    0.20100     |Sklearn MiniBatchKmeans|  Python   |
|       218.20000       |     15.51000     |     0.73370     |    0.01947     |         Knor          |     R     |

_________________________________________________________________________________________________________

## Release History

- 0.1.0 Initial release.
- 0.1.1 Added interface for MLJ.
- 0.1.2 Added `Elkan` algorithm.
- 0.1.3 Faster & optimized execution.
- 0.1.4 Bug fixes.
- 0.1.5 Added `Yinyang` algorithm.

## Contributing

Ultimately, we see this package as potentially the one-stop-shop for everything related to KMeans algorithm and its speed up variants. We are open to new implementations and ideas from anyone interested in this project.

Detailed contribution guidelines will be added in upcoming releases.

<!--- TODO: Contribution Guidelines --->

```@index
```

```@autodocs
Modules = [ParallelKMeans]
```
