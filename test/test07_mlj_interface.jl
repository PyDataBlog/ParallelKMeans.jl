module TestMLJInterface

using ParallelKMeans
using ParallelKMeans: KMeans
using Random
using Test
using Suppressor
using MLJBase


@testset "Test struct construction" begin
    model = KMeans()

    @test model.algo            == :Hamerly
    @test model.init            == nothing
    @test model.k               == 3
    @test model.k_init          == "k-means++"
    @test model.max_iters       == 300
    @test model.copy            == true
    @test model.threads         == Threads.nthreads()
    @test model.tol             == 1.0e-6
    @test model.verbosity       == 0
end


@testset "Test bad struct warings" begin
    @test_logs (:warn, "Unsupported KMeans variant, Defaulting to Hamerly algorithm.") ParallelKMeans.KMeans(algo=:Fake)
    @test_logs (:warn, "Only `k-means++` or random seeding algorithms are supported. Defaulting to k-means++ seeding.") ParallelKMeans.KMeans(k_init="abc")
    @test_logs (:warn, "Number of clusters must be greater than 0. Defaulting to 3 clusters.") ParallelKMeans.KMeans(k=0)
    @test_logs (:warn, "Tolerance level must be less than 1. Defaulting to tol of 1e-6.") ParallelKMeans.KMeans(tol=2)
    @test_logs (:warn, "Number of permitted iterations must be greater than 0. Defaulting to 300 iterations.") ParallelKMeans.KMeans(max_iters=0)
    @test_logs (:warn, "Number of threads must be at least 1. Defaulting to all threads available.") ParallelKMeans.KMeans(threads=0)
    @test_logs (:warn, "Verbosity must be either 0 (no info) or 1 (info requested). Defaulting to 1.") ParallelKMeans.KMeans(verbosity=100)
end


@testset "Test model fitting verbosity" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    model = KMeans(k=2, max_iters=1, verbosity=1)
    results = @capture_out fit(model, X)

    @test results == "Iteration 1: Jclust = 28.0\n"
end


@testset "Test Lloyd model fitting" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])

    model = KMeans(algo = :Lloyd, k=2)
    results = fit(model, X)

    @test results[2]             == nothing
    @test results[end].converged == true
    @test results[end].totalcost == 16

    params = fitted_params(model, results)
    @test params.converged == true
    @test params.totalcost == 16

    # Use trained model to cluster new data X_test
    preds = transform(model, results, X_test)
    @test preds[:x1][1] == 2
end


@testset "Test Hamerly model fitting" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])

    model = KMeans(algo=:Hamerly, k=2)
    results = fit(model, X)

    @test results[2]             == nothing
    @test results[end].converged == true
    @test results[end].totalcost == 16

    params = fitted_params(model, results)
    @test params.converged == true
    @test params.totalcost == 16

    # Use trained model to cluster new data X_test
    preds = transform(model, results, X_test)
    @test preds[:x1][1] == 2
end

@testset "Test Elkan model fitting" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])

    model = KMeans(algo=:Elkan, k=2)
    results = fit(model, X)

    @test results[2]             == nothing
    @test results[end].converged == true
    @test results[end].totalcost == 16

    params = fitted_params(model, results)
    @test params.converged == true
    @test params.totalcost == 16

    # Use trained model to cluster new data X_test
    preds = transform(model, results, X_test)
    @test preds[:x1][1] == 2
end

@testset "Testing non convergence warning" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])

    model = KMeans(k=2, max_iters=1)
    results = fit(model, X)

    @test_logs (:warn, "Failed to converged. Using last assignments to make transformations.") transform(model, results, X_test)
end

end # module
