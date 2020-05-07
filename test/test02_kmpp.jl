module TestKMPP

using ParallelKMeans
using ParallelKMeans: smart_init
using Test
using StableRNGs

# resulting indices obtained from `Clustering` implementation of kmpp
@testset "singlethread kmpp" begin
    rng = StableRNG(2020)
    design_matrix = rand(rng, 10, 1000)

    @test smart_init(design_matrix, 10, 1, nothing, rng).indices == [303, 123, 234, 773, 46, 312, 528, 124, 393, 910]
end

@testset "multithread kmpp" begin
    rng = StableRNG(2020)
    design_matrix = rand(rng, 10, 1000)

    @test smart_init(design_matrix, 10, 2, nothing, rng).indices == [303, 123, 234, 773, 46, 312, 528, 124, 393, 910]
end

end # module
