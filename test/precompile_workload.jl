using DifferentialEquations, EasyModelAnalysis, ModelingToolkit, Test

@testset "Precompile workload" begin
    @independent_variables t
    @variables x(t)
    D = Differential(t)
    @mtkcompile sys = System([D(x) ~ -x], t)
    prob = ODEProblem(sys, Dict(x => 1.0), (0.0, 1.0))

    @test length(get_timeseries(prob, x, [0.0, 0.5, 1.0])) == 3
end
