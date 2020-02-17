module TestParallelKMeans
using Test

for file in sort([file for file in readdir(@__DIR__) if
                                   occursin(r"^test[_0-9]+.*\.jl$", file)])
    m = match(r"test[_0-9]+(.*).jl", file)

    @testset "$(m[1])" begin
        # Here you can optionally exclude some test files
        # VERSION < v"1.1" && file == "test_xxx.jl" && continue

        include(file)
    end
end

end  # module
