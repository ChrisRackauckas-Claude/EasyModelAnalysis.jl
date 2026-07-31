module EasyModelAnalysis

# EasyModelAnalysis is a facade over the SciML modeling stack: `using EasyModelAnalysis`
# is meant to be enough on its own to build a model, solve it, sample a parameter
# distribution and plot the result. These four packages are brought into scope for that
# reexport surface, which is pinned by the explicit `export` blocks at the bottom of this
# file rather than by a blanket `@reexport`; every reexported name stays owned and
# documented by its upstream package.
using DifferentialEquations
using ModelingToolkit
using Distributions
using Plots

using LinearAlgebra: LinearAlgebra, I, norm
using DifferentialEquations: DifferentialEquations, remake, solve
using ModelingToolkit: ModelingToolkit, Num, Symbolics
using Distributions: Distributions, InverseGamma, MvNormal, product_distribution
using Plots: Plots, @layout, bar, plot, plot!, scatter!
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

# Reexported DifferentialEquations public API; approved via `reexports_allow` in
# test/qa/qa.jl.
export AbstractAnalyticalProblem, AddVector, AffineOperator, AllObserved, AnalyticalProblem,
    AutoDePSpecialize, AutoFiniteDiff, AutoForwardDiff, AutoSparse, AutoTsit5, AutoVern6,
    AutoVern7, AutoVern8, AutoVern9, BVPFunction, BVProblem, BatchIntegralFunction,
    BlockDiagonalOperator, CallbackSet, CheckInit, Clocks, ContinuousCallback,
    ConvexOptimizationProblem, DAEFunction, DAEProblem, DAESolution, DDEFunction,
    DDEProblem, DefaultODEAlgorithm, DiagonalOperator, DifferentialEquations,
    DiscreteCallback, DiscreteFunction, DiscreteProblem, DynamicalBVPFunction,
    DynamicalDDEFunction, DynamicalDDEProblem, DynamicalODEFunction, DynamicalODEProblem,
    DynamicalSDEFunction, DynamicalSDEProblem, EigenvalueProblem, EigenvalueSolution,
    EigenvalueTarget, EnsembleAnalysis, EnsembleContext, EnsembleDistributed,
    EnsembleProblem, EnsembleSerial, EnsembleSolution, EnsembleSplitThreads,
    EnsembleSummary, EnsembleTestSolution, EnsembleThreads, FBDF, FunctionOperator,
    HomotopyNonlinearFunction, HomotopyProblem, IdentityOperator, ImplicitDiscreteFunction,
    ImplicitDiscreteProblem, IncrementingODEFunction, IncrementingODEProblem,
    IntegralFunction, IntegralProblem, IntegralSolution, IntervalNonlinearFunction,
    IntervalNonlinearProblem, InvertibleOperator, LinearAliasSpecifier, LinearProblem,
    LinearSolution, MatrixOperator, MultiObjectiveOptimizationFunction, NoiseProblem,
    NonlinearFunction, NonlinearLeastSquaresProblem, NonlinearProblem, NonlinearSolution,
    NullOperator, ODEAliasSpecifier, ODEFunction, ODEInputFunction, ODEProblem, ODESolution,
    OptimizationFunction, OptimizationProblem, OptimizationSolution, OrdinaryDiffEq,
    PDENoTimeSolution, PDEProblem, PDETimeSeriesSolution, RODEFunction, RODEProblem,
    RODESolution, ReturnCode, Rodas5P, Rosenbrock23, SCCNonlinearProblem, SDDEFunction,
    SDDEProblem, SDEFunction, SDEProblem, SampledIntegralProblem, ScalarOperator, SciMLBase,
    SciMLLogging, SciMLOperators, SecondOrderBVProblem, SecondOrderDDEProblem,
    SecondOrderODEProblem, SplitFunction, SplitODEProblem, SplitSDEFunction,
    SplitSDEProblem, StaticWOperator, SteadyStateProblem, SteadyStateSolution,
    TensorProductOperator, TensorSumOperator, TimeDomain, Tsit5, TwoPointBVPFunction,
    TwoPointBVProblem, TwoPointDynamicalBVPFunction, TwoPointSecondOrderBVProblem,
    VectorContinuousCallback, Vern6, Vern7, Vern8, Vern9, WOperator, add_saveat!,
    add_tstop!, addat!, addat_non_user_cache!, addsteps!, auto_dt_reset!, cache_operator,
    change_t_via_interpolation!, check_error, check_keywords, concretize, deleteat!,
    deleteat_non_user_cache!, derivative_discontinuity!, discretize, du_cache, first_tstop,
    full_cache, get_dt, get_du, get_du!, get_proposed_dt, get_rng, get_tmp_cache,
    has_adjoint, has_concretization, has_exp, has_expmv, has_expmv!, has_ldiv, has_ldiv!,
    has_mul, has_mul!, has_rng, has_tstop, init, is_discrete_time_domain, iscached, isclock,
    isconstant, iscontinuous, isconvertible, isdiscrete, isinplace, islinear,
    issolverstepclock, issquare, kronsum, pop_tstop!, rand_cache, ratenoise_cache,
    reeval_internals_due_to_modification!, reinit!, remake, resize_non_user_cache!,
    savevalues!, set_abstol!, set_proposed_dt!, set_reltol!, set_rng!, set_t!, set_u!,
    solve, solve!, step!, successful_retcode, supports_solve_rng, symbolic_discretize,
    terminate!, u_cache, u_modified!, update_coefficients, update_coefficients!, user_cache,
    warn_compat

# Reexported ModelingToolkit public API; approved via `reexports_allow` in
# test/qa/qa.jl.
export @acrule, @arrayop, @brownian, @brownians, @component, @connector, @constants,
    @derivative_rule, @derivatives, @discretes, @independent_variables, @makearray,
    @mtkbuild, @mtkcompile, @mtkcomplete, @named, @namespace, @nonamespace, @pack!,
    @parameters, @poissonians, @register_array_symbolic, @register_derivative,
    @register_discontinuity, @register_inverse, @register_symbolic, @rule, @symbolic_wrap,
    @syms, @symstruct, @unpack, @variables, @wrapped, AbstractCollocation,
    AbstractDynamicOptProblem, AbstractNonlinearProblem, AnalysisPoint, AssignmentAffect,
    BS, BipartiteGraph, CasADiCollocation, CasADiDynamicOptProblem, Clock, CompilerOptions,
    Connection, Differential, DiscreteSystem, DynamicOptSolution, Equation, EvalAt, Flow,
    Girsanov_transform, GlobalScope, Hold, HomotopyContinuationProblem, IRStructure,
    ImplicitDiscreteSystem, Inequality, InfiniteOptCollocation,
    InfiniteOptDynamicOptProblem, Initial, InitializationProblem, Integral, JuMPCollocation,
    JuMPDynamicOptProblem, JumpProblem, JumpSystem, LinearizationOpPoint,
    LinearizationProblem, LocalScope, MTKParameters, MiscSystemData, MissingGuessValue,
    ModelingToolkit, ModelingToolkitBase, NonlinearSystem, Num, ODESystem,
    OptimizationSystem, PDESystem, ParentScope, Pre, PyomoCollocation,
    PyomoDynamicOptProblem, Rewriters, RuleSet, SDESystem, SafeReal, Sample, SampleTime,
    SemilinearODEFunction, SemilinearODEProblem, Shift, ShiftIndex, SolverStepClock, Stream,
    StructuralTransformations, SymReal, SymScope, SymStruct, SymbolicLinearODE,
    SymbolicMassActionJump, SymbolicUtils, Symbolics, SymbolicsSparsityDetector, System,
    TearingState, Term, TreeReal, UnPack, add_accumulations, alg_equations,
    alias_elimination, analytically_integrated, analyze_initialization_jacobian,
    approximation_function, arguments, asdigraph, asgraph, bindings, bound_parameters,
    brownians, build_function, but_ordered_incidence, calculate_control_jacobian,
    calculate_cost_gradient, calculate_cost_hessian, calculate_hessian, calculate_jacobian,
    calculate_massmatrix, calculate_tgrad, change_independent_variable, change_of_variables,
    complete, compose, connect, constraints, continuous_events, convert_system_indepvar,
    cost, dae_index_lowering, debug_system, diff_equations, discrete_events, domain_connect,
    dummy_derivative, eqeq_dependencies, equation_dependencies, equations, expand,
    expand_connections, expand_derivatives, extend, factors, find_solvables!, flatten,
    flatten_fractions, fractional_to_ordinary, full_equations, gather_factor, generate_W,
    generate_control_jacobian, generate_cost, generate_cost_gradient, generate_cost_hessian,
    generate_custom_function, generate_diffusion_function, generate_initializesystem,
    generate_jacobian, generate_rhs, generate_tgrad, get_alg_eqs, get_canonical_expr,
    get_comp_sensitivity, get_comp_sensitivity_function, get_diff_eqs, get_looptransfer,
    get_looptransfer_function, get_reachability, get_sensitivity_function, get_variables,
    getbounds, getconnect, getdescription, getdist, getguess, getmetadata, getmisc,
    getnominal, getunit, groebner_basis, guesses, has_alg_eqs, has_alg_equations,
    has_diff_eqs, has_diff_equations, has_inverse, has_left_inverse, has_right_inverse,
    hasbounds, hasconnect, hasdescription, hasdist, hasguess, hasmetadata, hasmisc,
    hasnominal, hasunit, hierarchy, highest_order_variable_mask, homotopy, ifelse_branching,
    ifelse_eager, independent_variable, independent_variables, infimum, initial_conditions,
    initialization_equations, instream, inverse, irreducibles, is_derivative,
    is_groebner_basis, iscall, isdisturbance, isinitial, isinput, isirreducible,
    isolate_subsystem, isoutput, istree, istunable, jumps, left_continuous_function,
    left_inverse, limit, linear_fractional_to_ordinary, linearization_ap_transform,
    linearization_function, linearize, liouville_transform, lowest_order_variable_mask,
    majorization_function, map_variables_to_equations, maybe_zeros, minorization_function,
    modelingtoolkitize, mtkcompile, mtkcompile!, noise_to_brownians, observables, observed,
    open_loop, operation, pantelides_reassemble, parameters, parse_expr_to_symbolic,
    polynomial_coeffs, populate_ir!, print_ir, quick_cancel, reorder_dimension_by_tunables,
    reorder_dimension_by_tunables!, respecialize, right_continuous_function, right_inverse,
    rootfunction, semilinear_form, semipolynomial_form, semiquadratic_form, series,
    set_defaults, setmetadata, setnominal, simplify, simplify_fractions, solve_for,
    solve_linear_ode_system, solve_symbolic_IVP, sorted_arguments, sorted_incidence_matrix,
    state_priorities, state_priority, stochastic_integral_transform, structural_simplify,
    subset_tunables, substitute, substitute_component, substitute_in_deriv,
    substitute_in_deriv_and_depvar, supremum, symbolic_linear_solve, symbolic_solve,
    symbolic_solve_ode, symbolics_to_sympy, symbolics_to_sympy_pythoncall,
    sympy_algebraic_solve, sympy_integrate, sympy_limit, sympy_linear_solve,
    sympy_ode_solve, sympy_pythoncall_algebraic_solve, sympy_pythoncall_integrate,
    sympy_pythoncall_limit, sympy_pythoncall_linear_solve, sympy_pythoncall_ode_solve,
    sympy_pythoncall_simplify, sympy_pythoncall_to_symbolics, sympy_simplify,
    sympy_to_symbolics, taylor, taylor_coeff, tearing, tearing_substitution, term, terms,
    toexpr, toggle_namespacing, tosymbol, tunable_parameters, unknowns, unwrap_const,
    variable_dependencies, vartype, varvar_dependencies, ≲, ≳

# Reexported Distributions public API; approved via `reexports_allow` in
# test/qa/qa.jl.
export AbstractMixtureModel, AbstractMvNormal, Arcsine, ArrayLikeVariate, Bernoulli,
    BernoulliLogit, Beta, BetaBinomial, BetaPrime, Binomial, Biweight, Categorical, Cauchy,
    Chernoff, Chi, Chisq, CholeskyVariate, Continuous, ContinuousDistribution,
    ContinuousMatrixDistribution, ContinuousMultivariateDistribution,
    ContinuousUnivariateDistribution, Cosine, DiagNormal, DiagNormalCanon, Dirac, Dirichlet,
    DirichletMultinomial, Discrete, DiscreteDistribution, DiscreteMatrixDistribution,
    DiscreteMultivariateDistribution, DiscreteNonParametric, DiscreteUniform,
    DiscreteUnivariateDistribution, Distribution, Distributions, DoubleExponential,
    EdgeworthMean, EdgeworthSum, EdgeworthZ, Epanechnikov, Erlang, Estimator, Exponential,
    FDist, FisherNoncentralHypergeometric, Frechet, FullNormal, FullNormalCanon, Gamma,
    GeneralizedExtremeValue, GeneralizedPareto, Geometric, Gumbel, Hypergeometric,
    InverseGamma, InverseGaussian, InverseWishart, IsoNormal, IsoNormalCanon, JohnsonSU,
    JointOrderStatistics, KSDist, KSOneSided, Kolmogorov, Kumaraswamy, LKJ, LKJCholesky,
    Laplace, Levy, Lindley, LocationScale, LogLogistic, LogNormal, LogUniform, Logistic,
    LogitNormal, MLEstimator, MatrixBeta, MatrixDistribution, MatrixFDist, MatrixNormal,
    MatrixReshaped, MatrixTDist, Matrixvariate, MixtureModel, Multinomial, Multivariate,
    MultivariateDistribution, MultivariateMixture, MultivariateNormal, MvLogNormal,
    MvLogitNormal, MvNormal, MvNormalCanon, MvNormalKnownCov, MvTDist, NamedTupleVariate,
    NegativeBinomial, NonMatrixDistribution, NoncentralBeta, NoncentralChisq, NoncentralF,
    NoncentralHypergeometric, NoncentralT, Normal, NormalCanon, NormalInverseGaussian,
    OrderStatistic, PGeneralizedGaussian, Pareto, Poisson, PoissonBinomial, Product, QQPair,
    Rayleigh, RealInterval, Rician, Sampleable, Semicircle, Skellam, SkewNormal,
    SkewedExponentialPower, Soliton, StudentizedRange, SufficientStats, SymTriangularDist,
    TDist, TriangularDist, Triweight, Truncated, Uniform, Univariate,
    UnivariateDistribution, UnivariateGMM, UnivariateMixture, ValueSupport, VariateForm,
    VonMises, VonMisesFisher, WalleniusNoncentralHypergeometric, Weibull, Wishart,
    ZeroMeanDiagNormal, ZeroMeanDiagNormalCanon, ZeroMeanFullNormal,
    ZeroMeanFullNormalCanon, ZeroMeanIsoNormal, ZeroMeanIsoNormalCanon, canonform, ccdf,
    cdf, censored, cf, cgf, circvar, component, components, componentwise_logpdf,
    componentwise_pdf, concentration, convolve, cor, cov, cquantile, dim, dof, entropy,
    estimate, expected_logdet, failprob, fit, fit_mle, gradlogpdf, hasfinitesupport,
    insupport, invcov, invlogccdf, invlogcdf, invscale, isbounded, isleptokurtic,
    islowerbounded, ismesokurtic, isplatykurtic, isprobvec, isupperbounded, kldivergence,
    kurtosis, location, location!, logccdf, logcdf, logdetcov, logdiffcdf, loglikelihood,
    logpdf, logpdf!, mean, meandir, meanform, meanlogx, median, mgf, mode, modes, moment,
    ncategories, ncomponents, nsamples, ntrials, params, params!, partype, pdf,
    pdfsquaredL2norm, probs, probval, product_distribution, qqbuild, quantile, rate, sample,
    sample!, sampler, scale, scale!, shape, skewness, span, sqmahal, sqmahal!, std, stdlogx,
    succprob, suffstats, support, truncated, var, varlogx, wsample, wsample!

# Reexported Plots public API; approved via `reexports_allow` in
# test/qa/qa.jl.
export @P_str, @animate, @colorant_str, @gif, @layout, @recipe, @series, @shorthands,
    @userplot, ABGR, ADIN99, ADIN99d, ADIN99o, AGray, AGray32, AHSI, AHSL, AHSV, ALCHab,
    ALCHuv, ALMS, ALab, ALuv, AOklab, AOklch, ARGB, ARGB32, AXYZ, AYCbCr, AYIQ,
    AbstractAGray, AbstractARGB, AbstractBackend, AbstractGray, AbstractGrayA,
    AbstractLayout, AbstractPlot, AbstractRGB, AbstractRGBA, AlphaColor, Animation, AxyY,
    BGR, BGRA, BezierCurve, CIE1931JV_CMF, CIE1931J_CMF, CIE1931_CMF, CIE1964_CMF,
    CIE2006_10_CMF, CIE2006_2_CMF, Color, Color3, ColorAlpha, ColorGradient, ColorPalette,
    ColorTypes, Colorant, ColorantNormed, Colors, DE_2000, DE_94, DE_AB, DE_BFD, DE_CMC,
    DE_DIN99, DE_DIN99d, DE_DIN99o, DE_JPC79, DIN99, DIN99A, DIN99d, DIN99dA, DIN99o,
    DIN99oA, Formatted, Fractional, GR, Gray, Gray24, GrayA, HSB, HSI, HSIA, HSL, HSLA, HSV,
    HSVA, KW, LCHab, LCHabA, LCHuv, LCHuvA, LMS, LMSA, Lab, LabA, Luv, LuvA, MSC, OHLC,
    Oklab, OklabA, Oklch, OklchA, PlotTheme, PlotThemes, PlotUtils, Plots, RGB, RGB24, RGBA,
    RGBX, RecipeData, RecipesBase, Segments, Shape, Surface, Transparent3, TransparentColor,
    TransparentGray, TransparentRGB, XRGB, XYZ, XYZA, YCbCr, YCbCrA, YIQ, YIQA,
    adapted_grid, add_theme, aliases, alpha, alphacolor, animate, annotate!, areaplot,
    areaplot!, arrow, attr!, backend, backend_name, backend_object, backends, bar, bar!,
    barh, barh!, barhist, barhist!, base_color_type, base_colorant_type, bbox, blue,
    boxplot, boxplot!, brush, ccolor, center, cgrad, chroma, cie_color_match, closeall,
    color, color_list, color_type, coloralpha, colordiff, colormap, colormatch, comp1,
    comp2, comp3, comp4, comp5, contour, contour!, contour3d, contour3d!, contourf,
    contourf!, coords, current, curve_points, curves, curves!, cvec, default, default_cgrad,
    density, density!, deuteranopic, distinguishable_colors, diverging_palette, font, frame,
    gamutmax, gamutmin, gaston, get_color_palette, gif, gr, gray, green, grid, gui, hdf5,
    heatmap, heatmap!, hex, hexbin, hexbin!, histogram, histogram!, histogram2d,
    histogram2d!, hline, hline!, hspan, hspan!, hue, inline, inspectdr, invisible, isdark,
    iter_segments, lens!, mapc, mapreducec, mean_hue, mesh3d, mesh3d!, mov, mp4,
    normalize_hue, ohlc, ohlc!, optimize_datetime_ticks, optimize_ticks, palette,
    parametric_colorant, path3d, path3d!, pgfplots, pgfplotsx, pie, pie!, plot, plot!,
    plot3d, plot3d!, plot_color, plotarea, plotattr, plotly, plotlyjs, plots_heatmap,
    plots_heatmap!, png, portfoliocomposition, portfoliocomposition!, protanopic, pyplot,
    pythonplot, quiver, quiver!, red, reducec, resetfontsizes, rgb_string, rgba_string,
    rotate, rotate!, savefig, scalefontsize, scalefontsizes, scatter, scatter!, scatter3d,
    scatter3d!, scatterhist, scatterhist!, sequential_palette, set_theme, shape_coords,
    showtheme, showtheme!, spy, spy!, stephist, stephist!, sticks, sticks!, stroke, surface,
    surface!, test_examples, text, theme, theme_palette, title!, translate, translate!,
    tritanopic, twinx, twiny, unicodeplots, violin, violin!, vline, vline!, vspan, vspan!,
    webm, weighted_color_mean, whitebalance, wireframe, wireframe!, with, wrap, xaxis!,
    xerror, xerror!, xflip!, xgrid!, xlabel!, xlims, xlims!, xticks, xticks!, xyY, xyYA,
    yaxis!, yerror, yerror!, yflip!, ygrid!, ylabel!, ylims, ylims!, yticks, yticks!,
    zaxis!, zerror, zerror!, zflip!, zgrid!, zlabel!, zlims, zlims!, zscale, zticks, zticks!

end
