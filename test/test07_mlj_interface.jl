module TestMLJInterface

using MLJModelInterface
using ParallelKMeans
using Random
using Test
using Suppressor
using MLJBase


@testset "Test struct construction" begin
    model = ParallelKMeans.KMeans()

    @test model.algo            == Lloyd()
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

end


@testset "Test fitted params" begin
    
end


@testset "Test transform" begin
    
end


end # end module