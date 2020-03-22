module TestKMPP

using ParallelKMeans
using ParallelKMeans: MultiThread
using Test
using Random

# resulting indices obtained from `Clustering` implementation of kmpp
@testset "singlethread kmpp" begin
    Random.seed!(1980)
    design_matrix = rand(10, 1000)

    Random.seed!(2020)
    @test ParallelKMeans.smart_init(design_matrix, 10).indices == [33, 931, 853, 940, 926, 528, 644, 552, 460, 433]
end

@testset "multithread kmpp" begin
    Random.seed!(1980)
    design_matrix = rand(10, 1000)

    Random.seed!(2020)
    @test ParallelKMeans.smart_init(design_matrix, 10, MultiThread(2)).indices == [33, 931, 853, 940, 926, 528, 644, 552, 460, 433]
end

end # module
