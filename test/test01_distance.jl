module TestDistance
using ParallelKMeans: pairwise!, SingleThread, MultiThread
using Test

@testset "naive singlethread pairwise" begin
    X = [1.0 2.0; 3.0 5.0; 4.0 6.0]
    y = [1.0 2.0; ]
    r = Array{Float64, 2}(undef, 3, 1)

    pairwise!(r, X, y)
    @test all(r .≈ [0.0, 13.0, 25.0])
end

@testset "multithread pairwise" begin
    X = [1.0 2.0; 3.0 5.0; 4.0 6.0]
    y = [1.0 2.0; ]
    r = Array{Float64, 2}(undef, 3, 1)

    pairwise!(r, X, y, MultiThread())
    @test all(r .≈ [0.0, 13.0, 25.0])
end


end # module
