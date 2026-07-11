using SciMLTesting, EasyModelAnalysis, Test

run_qa(
    EasyModelAnalysis;
    explicit_imports = true,
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    api_docs_kwargs = (;
        ignore = (
            :EnsembleAnalysis,
            # Non-owned reexports from the modeling/plotting/statistics stack.
            Symbol("@brownian"), Symbol("@mtkbuild"), Symbol("@symbolic_wrap"),
            Symbol("@wrapped"), :ABGR, :ADIN99, :ADIN99d, :ADIN99o, :AGray,
            :AHSI, :AHSL, :AHSV, :ALCHab, :ALCHuv, :ALMS, :ALab, :ALuv,
            :AOklab, :AOklch, :ARGB, :AXYZ, :AYCbCr, :AYIQ, :AbstractAGray,
            :AbstractARGB, :AbstractBackend, :AbstractCollocation, :AbstractGrayA,
            :AbstractLayout, :AbstractPlot, :AbstractRGBA, :AxyY, :BGRA,
            :CIE1931JV_CMF, :CIE1931J_CMF, :CIE1931_CMF, :CIE1964_CMF,
            :CIE2006_10_CMF, :CIE2006_2_CMF, :Color3, :ColorGradient,
            :ColorPalette, :ColorantNormed, :ContinuousDistribution, :DIN99A,
            :DIN99dA, :DIN99oA, :DefaultODEAlgorithm, :DiagNormal,
            :DiagNormalCanon, :DiscreteDistribution, :DiscreteSystem,
            :DoubleExponential, :DynamicOptSolution, :EdgeworthMean, :EdgeworthSum,
            :EdgeworthZ, :Estimator, :FisherNoncentralHypergeometric, :Fractional,
            :FullNormal, :FullNormalCanon, :GrayA, :HSIA, :HSLA, :HSVA,
            :ImplicitDiscreteSystem, :IsoNormal, :IsoNormalCanon, :LCHabA,
            :LCHuvA, :LMSA, :LabA, :LocationScale, :LuvA, :MLEstimator,
            :MatrixDistribution, :MatrixReshaped, :MultivariateDistribution,
            :MultivariateMixture, :MultivariateNormal, :MvNormalKnownCov, :MvTDist,
            :NonMatrixDistribution, :NoncentralHypergeometric, :ODESystem, :OklabA,
            :OklchA, :PlotTheme, :QQPair, :RGBA, :RealInterval, :RecipeData,
            :RuleSet, :SufficientStats, :Transparent3, :TransparentGray,
            :TransparentRGB, :UnivariateDistribution, :UnivariateGMM,
            :UnivariateMixture, :VonMisesFisher, :WalleniusNoncentralHypergeometric,
            :XYZA, :YCbCrA, :YIQA, :ZeroMeanDiagNormal, :ZeroMeanDiagNormalCanon,
            :ZeroMeanFullNormal, :ZeroMeanFullNormalCanon, :ZeroMeanIsoNormal,
            :ZeroMeanIsoNormalCanon, :add_theme, :alias_elimination, :aliases,
            :areaplot!, :attr!, :barh, :barh!, :brush, :but_ordered_incidence,
            :canonform, :cie_color_match, :circvar, :color_list, :component,
            :componentwise_logpdf, :componentwise_pdf, :concentration, :contour3d,
            :contour3d!, :curve_points, :cvec, :default_cgrad, :dim, :estimate,
            :expected_logdet, :gamutmax, :gamutmin, :gaston, :get_canonical_expr,
            :get_color_palette, :gr, :hasfinitesupport, :hdf5,
            :highest_order_variable_mask, :independent_variable, :infimum, :inline,
            :inspectdr, :invisible, :invscale, :irreducibles, :is_derivative,
            :isbounded, :isdark, :islowerbounded, :isprobvec, :istree,
            :isupperbounded, :iter_segments, :lowest_order_variable_mask,
            :maybe_zeros, :meandir, :meanform, :meanlogx, :mtkcompile!,
            :ncomponents, :optimize_datetime_ticks, :pantelides_reassemble,
            :partype, :pgfplots, :pgfplotsx, :plot!, :plot3d!, :plot_color,
            :plotly, :plotlyjs, :plots_heatmap, :plots_heatmap!,
            :portfoliocomposition, :portfoliocomposition!, :probval, :pyplot,
            :pythonplot, :qqbuild, :resetfontsizes, :rgb_string, :rgba_string,
            :rotate!, :scalefontsize, :set_theme, :setnominal, :shape_coords,
            :showtheme, :showtheme!, :solve_for, :spy, :spy!, :sqmahal!, :stdlogx,
            :structural_simplify, :suffstats, :supremum, :tearing_substitution,
            :translate!, :unicodeplots, :varlogx, :wrap, :xyYA,
        ),
    ),
    # undefined_exports: `Variable` and `rotate!` leak in (dead) via `@reexport`
    # (SciML/EasyModelAnalysis.jl#300)
    aqua_broken = (:undefined_exports,),
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (;
            ignore = (
                # `value` is the Symbolics unwrap API, re-exported through ModelingToolkit
                # (the direct dependency); accessing it as `ModelingToolkit.value` is intended.
                :value,
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractMCMCEnsemble,  # AbstractMCMC: no public alias for the ensemble type
                :LN_SBPLX,              # NLopt: algorithm constant, not marked public
                :value,                 # ModelingToolkit: Symbolics unwrap re-export, not marked public
            ),
        ),
    ),
)
