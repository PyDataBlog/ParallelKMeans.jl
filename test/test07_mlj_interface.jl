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
end


@testset "Test bad struct warings" begin
    @test_logs (:warn, "Unsupported KMeans variant. Defaulting to Hamerly algorithm.") ParallelKMeans.KMeans(algo=:Fake)
    @test_logs (:warn, "Only \"k-means++\" or \"random\" seeding algorithms are supported. Defaulting to k-means++ seeding.") ParallelKMeans.KMeans(k_init="abc")
    @test_logs (:warn, "Number of clusters must be greater than 0. Defaulting to 3 clusters.") ParallelKMeans.KMeans(k=0)
    @test_logs (:warn, "Tolerance level must be less than 1. Defaulting to tol of 1e-6.") ParallelKMeans.KMeans(tol=2)
    @test_logs (:warn, "Number of permitted iterations must be greater than 0. Defaulting to 300 iterations.") ParallelKMeans.KMeans(max_iters=0)
    @test_logs (:warn, "Number of threads must be at least 1. Defaulting to all threads available.") ParallelKMeans.KMeans(threads=0)
end


@testset "Test model fitting verbosity" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    model = KMeans(k=2, max_iters=1)
    results = @capture_out fit(model, 1, X)

    @test results == "Iteration 1: Jclust = 28.0\n"
end


@testset "Test Lloyd model fitting" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])

    model = KMeans(algo = :Lloyd, k=2)
    results, cache, report = fit(model, 0, X)

    @test cache             == nothing
    @test report.converged  == true
    @test report.totalcost  == 16

    params = fitted_params(model, results)
    @test params.cluster_centers == [1.0 10.0; 2.0 2.0]

    # Use trained model to cluster new data X_test
    preds = transform(model, results, X_test)
    @test preds[:x1][1] == 82.0
    @test preds[:x2][1] == 1.0

    # Make predictions on new data X_test with fitted params
    yhat = predict(model, results, X_test)
    @test yhat[1] == 2
end


@testset "Test Hamerly model fitting" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])

    model = KMeans(algo = :Hamerly, k=2)
    results, cache, report = fit(model, 0, X)

    @test cache             == nothing
    @test report.converged  == true
    @test report.totalcost  == 16

    params = fitted_params(model, results)
    @test params.cluster_centers == [1.0 10.0; 2.0 2.0]

    # Use trained model to cluster new data X_test
    preds = transform(model, results, X_test)
    @test preds[:x1][1] == 82.0
    @test preds[:x2][1] == 1.0

    # Make predictions on new data X_test with fitted params
    yhat = predict(model, results, X_test)
    @test yhat[1] == 2
end


@testset "Test Elkan model fitting" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])

    model = KMeans(algo = :Elkan, k=2)
    results, cache, report = fit(model, 0, X)

    @test cache             == nothing
    @test report.converged  == true
    @test report.totalcost  == 16

    params = fitted_params(model, results)
    @test params.cluster_centers == [1.0 10.0; 2.0 2.0]

    # Use trained model to cluster new data X_test
    preds = transform(model, results, X_test)
    @test preds[:x1][1] == 82.0
    @test preds[:x2][1] == 1.0

    # Make predictions on new data X_test with fitted params
    yhat = predict(model, results, X_test)
    @test yhat[1] == 2
end


@testset "Testing non convergence warning" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])

    model = KMeans(k=2, max_iters=1)
    results, cache, report = fit(model, 0, X)

    @test_logs (:warn, "Failed to converge. Using last assignments to make transformations.") transform(model, results, X_test)
end

"""
@testset "Testing non convergence warning during model fitting" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])

    model = KMeans(k=2, max_iters=1)
    @test_logs (:warn, "Specified model failed to converge.") fit(model, 1, X);
end
"""
end # module
