module EasyModelAnalysis

using LinearAlgebra: LinearAlgebra, I, norm
using DifferentialEquations: DifferentialEquations, remake, solve
using ModelingToolkit: ModelingToolkit, Num, Symbolics
using Distributions: Distributions, InverseGamma, MvNormal, product_distribution
using Plots: Plots, @layout, bar, plot, plot!, scatter!
using PrecompileTools: @compile_workload, @setup_workload
using Optimization: Optimization, OptimizationProblem
using OptimizationBBO: OptimizationBBO, BBO_adaptive_de_rand_1_bin_radiuslimited
using OptimizationNLopt: OptimizationNLopt
using GlobalSensitivity: GlobalSensitivity, Sobol
using NLopt: NLopt, Opt, inequality_constraint!
using Turing: Turing, @varname
using Integrals: Integrals, HCubatureJL
using SciMLExpectations: SciMLExpectations, ExpectationProblem, GenericDistribution,
    Koopman, SystemMap
using SciMLBase: SciMLBase, ContinuousCallback, EnsembleProblem, EnsembleSerial,
    EnsembleSolution, EnsembleThreads, ODESolution, terminate!
using SymbolicUtils: SymbolicUtils

include("basics.jl")
include("datafit.jl")
include("sensitivity.jl")
include("threshold.jl")
include("intervention.jl")
include("ensemble.jl")

export get_timeseries, get_min_t, get_max_t, plot_extrema, phaseplot_extrema
export get_uncertainty_forecast, get_uncertainty_forecast_quantiles
export plot_uncertainty_forecast, plot_uncertainty_forecast_quantiles
export datafit, global_datafit, bayesian_datafit
export get_sensitivity, create_sensitivity_plot, get_sensitivity_of_maximum
export stop_at_threshold, get_threshold
export model_forecast_score
export optimal_threshold_intervention, prob_violating_threshold,
    optimal_parameter_intervention_for_threshold, optimal_parameter_threshold,
    optimal_parameter_intervention_for_reach
export bayesian_ensemble, ensemble_weights

@setup_workload begin
    @compile_workload begin
        ModelingToolkit.@independent_variables t
        ModelingToolkit.@variables x(t)
        D = ModelingToolkit.Differential(t)
        ModelingToolkit.@mtkcompile sys = ModelingToolkit.System([D(x) ~ -x], t)
        prob = DifferentialEquations.ODEProblem(sys, Dict(x => 1.0), (0.0, 1.0))
        get_timeseries(prob, x, [0.0, 0.5, 1.0])
    end
end

end
