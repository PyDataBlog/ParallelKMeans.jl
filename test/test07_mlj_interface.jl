module TestMLJInterface

using ParallelKMeans
using Random
using Test
using Suppressor
using MLJBase


@testset "Test struct construction" begin
    model = KMeans()

    @test model.algo            == :Lloyd
    @test model.init            == nothing
    @test model.k               == 3
    @test model.k_init          == "k-means++"
    @test model.max_iters       == 300
    @test model.copy            == true
    @test model.threads         == Threads.nthreads()
    @test model.tol             == 1.0e-6
    @test model.verbosity       == 0
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
    model = KMeans(k=2)
    results = fit(model, X)

    @test results[2]             == nothing
    @test results[end].converged == true
    @test results[end].totalcost == 16
end


@testset "Test Hamerly model fitting" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    model = KMeans(algo=:Hamerly, k=2)
    results = fit(model, X)

    @test results[2]             == nothing
    @test results[end].converged == true
    @test results[end].totalcost == 16
end


@testset "Test Lloyd fitted params" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    model = KMeans(k=2)
    results = fit(model, X)

    params = fitted_params(model, results)
    @test params.converged == true
    @test params.totalcost == 16
end


@testset "Test Hamerly fitted params" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    model = KMeans(algo=:Hamerly, k=2)
    results = fit(model, X)

    params = fitted_params(model, results)
    @test params.converged == true
    @test params.totalcost == 16
end


@testset "Test Lloyd transform" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])

    # Train model using training data X
    model = KMeans(k=2)
    results = fit(model, X)

    # Use trained model to cluster new data X_test
    preds = transform(model, results, X_test)
    @test preds[:x1][1] == 2
end


@testset "Test Hamerly transform" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])

    # Train model using training data X
    model = KMeans(algo=:Hamerly, k=2)
    results = fit(model, X)

    # Use trained model to cluster new data X_test
    preds = transform(model, results, X_test)
    @test preds[:x1][1] == 2
end

end # module