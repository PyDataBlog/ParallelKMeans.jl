module TestDistance
using ParallelKMeans: colwise!, SingleThread, MultiThread
using Test

@testset "naive singlethread colwise" begin
    X = [1.0 3.0 4.0; 2.0 5.0 6.0]
    y = [1.0, 2.0]
    r = Vector{Float64}(undef, 3)

    colwise!(r, X, y)
    @test all(r .≈ [0.0, 13.0, 25.0])
end

@testset "multithread colwise" begin
    X = [1.0 3.0 4.0; 2.0 5.0 6.0]
    y = [1.0, 2.0]
    r = Vector{Float64}(undef, 3)

    colwise!(r, X, y, MultiThread())
    @test all(r .≈ [0.0, 13.0, 25.0])
end

end # module
