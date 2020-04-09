module TestMLJInterface

using MLJModelInterface
using ParallelKMeans
using Random
using Test
using Suppressor
using MLJBase


@testset "Test struct construction" begin
    model = ParallelKMeans.KMeans()

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


@testset "Test model fitting" begin
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    model = ParallelKMeans.KMeans(k=2)
    results = fit(model, X)

    @test results[2]             == nothing
    @test results[end].converged == true
    @test results[end].totalcost == 16
end


@testset "Test fitted params" begin
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    model = ParallelKMeans.KMeans(k=2)
    results = fit(model, X)

    params = fitted_params(model, results)
    @test params.converged == true
    @test params.totalcost == 16
    
end


@testset "Test transform" begin
    Random.seed!(2020)
    X = table([1 2; 1 4; 1 0; 10 2; 10 4; 10 0])
    X_test = table([10 1])
    
    # Train model using training data X
    model = ParallelKMeans.KMeans(k=2)
    results = fit(model, X)

    # Use trained model to cluster new data X_test
    preds = transform(model, results, X_test)
    @test preds[:x1][1] == 2
end


end # end module