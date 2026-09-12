using EasyModelAnalysis, BenchmarkTools
using DifferentialEquations, Distributions, ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

const SUITE = BenchmarkGroup()

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [
    D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
]

@mtkbuild sys = ODESystem(eqs, t)

u0 = [D(x) => 2.0, x => 1.0, y => 0.0, z => 0.0]
p = [σ => 28.0, ρ => 10.0, β => 8 / 3]
tspan = (0.0, 50.0)
prob = ODEProblem(sys, u0, tspan, p; jac = true)
sol = solve(prob)

t_measure = [0.0, 1.0, 2.0, 5.0, 10.0]

# =============================================================================
# Timeseries extraction and extremes
# =============================================================================

SUITE["timeseries"] = BenchmarkGroup()

SUITE["timeseries"]["get_timeseries"] = @benchmarkable get_timeseries(
    $prob, $x, $t_measure
)
SUITE["timeseries"]["get_min_t"] = @benchmarkable get_min_t($prob, $x)
SUITE["timeseries"]["get_max_t"] = @benchmarkable get_max_t($prob, $x)
SUITE["timeseries"]["stop_at_threshold"] = @benchmarkable stop_at_threshold(
    $prob, $x, 5.0
)

# =============================================================================
# Sensitivity analysis (Morris over parameter bounds)
# =============================================================================

SUITE["sensitivity"] = BenchmarkGroup()

pbounds = [ρ => [0.0, 20.0], β => [0.0, 10.0]]
SUITE["sensitivity"]["morris"] = @benchmarkable EasyModelAnalysis.get_sensitivity(
    $prob, 40.0, $y, $pbounds; samples = 100
)
