"""
    get_timeseries(prob, sym, t) -> AbstractVector

Solve `prob` and return the values of the symbolic state or observable `sym` at `t`.

# Arguments

  - `prob`: a SciML problem with a finite `tspan` and symbolic indexing support.
  - `sym`: the symbolic state or observable to evaluate.
  - `t`: ordered evaluation times within `prob.tspan`.

# Returns

  - An array containing `sym` evaluated at every requested time.

# Examples

```julia
values = get_timeseries(prob, x, 0.0:0.1:10.0)
```
"""
function get_timeseries(prob, sym, t)
    @assert t[1] >= prob.tspan[1]
    prob = remake(prob, tspan = (prob.tspan[1], min(prob.tspan[2], t[end])))
    sol = solve(prob, saveat = t)
    return sol[sym]
end

"""
    get_min_t(prob, sym)
    get_min_t(sol, sym)

Find the minimum of `sym` over the problem or solution time span.

# Arguments

  - `prob`: a SciML problem to solve, or an existing solution.
  - `sym`: the symbolic state or observable to minimize.

# Returns

  - `(t, value)`, where `t` is the minimizing time and `value` is the corresponding value of
    `sym`.

# Examples

```julia
tmin, xmin = get_min_t(prob, x)
```
"""
function get_min_t(prob, sym)
    if prob isa ODESolution
        sol = prob
        prob = sol.prob
    else
        sol = solve(prob)
    end
    f(t, _) = sol(t[1]; idxs = sym)
    oprob = OptimizationProblem(
        f, [(prob.tspan[2] - prob.tspan[1]) / 2],
        lb = [prob.tspan[1]],
        ub = [prob.tspan[end]]
    )
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10000)
    return res.u[1], f(res.u[1], nothing)
end

"""
    get_max_t(prob, sym)
    get_max_t(sol, sym)

Find the maximum of `sym` over the problem or solution time span.

# Arguments

  - `prob`: a SciML problem to solve, or an existing solution.
  - `sym`: the symbolic state or observable to maximize.

# Returns

  - `(t, value)`, where `t` is the maximizing time and `value` is the corresponding value of
    `sym`.

# Examples

```julia
tmax, xmax = get_max_t(prob, x)
```
"""
function get_max_t(prob, sym)
    if prob isa ODESolution
        sol = prob
        prob = sol.prob
    else
        sol = solve(prob)
    end
    f(t, _) = -sol(t[1]; idxs = sym)
    oprob = OptimizationProblem(
        f, [(prob.tspan[2] - prob.tspan[1]) / 2],
        lb = [prob.tspan[1]],
        ub = [prob.tspan[end]]
    )
    res = solve(oprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10000)
    return res.u[1], -f(res.u[1], nothing)
end

"""
    plot_extrema(prob, sym) -> Plots.Plot

Plot `sym` and mark its minimum and maximum over `prob.tspan`.

# Arguments

  - `prob`: a SciML problem to solve.
  - `sym`: the symbolic state or observable to plot.

# Returns

  - The plot with extrema markers added.

# Examples

```julia
p = plot_extrema(prob, x)
```
"""
function plot_extrema(prob, sym)
    xmin, xminval = get_min_t(prob, sym)
    xmax, xmaxval = get_max_t(prob, sym)
    sol = solve(prob)
    plot(sol, idxs = sym)
    scatter!([xmin], [xminval])
    return scatter!([xmax], [xmaxval])
end

"""
    phaseplot_extrema(prob, sym, plotsyms) -> Plots.Plot

Plot a phase portrait and mark the points where `sym` reaches its extrema.

# Arguments

  - `prob`: a SciML problem to solve.
  - `sym`: the symbolic state or observable whose extrema are marked.
  - `plotsyms`: a tuple of symbolic states or observables defining the phase-plot axes.

# Returns

  - The phase plot with extrema markers added.

# Examples

```julia
p = phaseplot_extrema(prob, x, (x, y))
```
"""
function phaseplot_extrema(prob, sym, plotsyms)
    sol = solve(prob)
    xmin, xminval = get_min_t(prob, sym)
    xmax, xmaxval = get_max_t(prob, sym)
    plot(sol, idxs = plotsyms)
    scatter!([[sol(xmin; idxs = x)] for x in plotsyms]...)
    return scatter!([[sol(xmax; idxs = x)] for x in plotsyms]...)
end

"""
    get_uncertainty_forecast(prob, sym, t, uncertainp, samples) -> Vector

Return sampled trajectories of `sym` under independently distributed uncertain parameters.

# Arguments

  - `prob`: a SciML problem to simulate.
  - `sym`: the symbolic state or observable to collect from each trajectory.
  - `t`: ordered saved times within `prob.tspan`.
  - `uncertainp`: parameter-to-distribution pairs such as `[k => Uniform(0.8, 1.2)]`.
  - `samples::Integer`: number of trajectories to draw.

# Returns

  - A vector containing the sampled time series, one entry per trajectory.

# Examples

```julia
forecast = get_uncertainty_forecast(prob, x, 0.0:0.1:10.0, [k => Uniform(0.8, 1.2)], 100)
```
"""
function get_uncertainty_forecast(prob, sym, t, uncertainp, samples)
    @assert t[1] >= prob.tspan[1]
    function sample_prob(prob)
        ps = getindex.(uncertainp, 1) .=> rand.(getindex.(uncertainp, 2))
        return prob = remake(
            prob, tspan = (prob.tspan[1], min(prob.tspan[2], t[end])),
            p = ps
        )
    end
    prob_func(prob, i, repeat) = sample_prob(prob)
    prob_func(prob, ctx) = sample_prob(prob)
    eprob = EnsembleProblem(prob, prob_func = prob_func)
    esol = solve(eprob, nothing, EnsembleSerial(), saveat = t, trajectories = samples)
    return Array.(reduce.(hcat, [esol.u[i][sym] for i in 1:samples]))
end

"""
    get_uncertainty_forecast_quantiles(
            prob, sym, t, uncertainp, samples,
            quants = (0.05, 0.95)
        ) -> Vector

Estimate pointwise quantiles of `sym` from an uncertain-parameter ensemble.

# Arguments

  - `prob`: a SciML problem to simulate.
  - `sym`: the symbolic state or observable to collect.
  - `t`: ordered saved times within `prob.tspan`.
  - `uncertainp`: parameter-to-distribution pairs.
  - `samples::Integer`: number of ensemble trajectories.
  - `quants`: quantile probabilities to estimate.

# Returns

  - A vector of time series, one for each entry of `quants`.

# Examples

```julia
lower, upper = get_uncertainty_forecast_quantiles(
    prob, x, 0.0:0.1:10.0, [k => Uniform(0.8, 1.2)], 100
)
```
"""
function get_uncertainty_forecast_quantiles(
        prob, sym, t, uncertainp, samples,
        quants = (0.05, 0.95)
    )
    @assert t[1] >= prob.tspan[1]
    function sample_prob(prob)
        ps = getindex.(uncertainp, 1) .=> rand.(getindex.(uncertainp, 2))
        return prob = remake(
            prob, tspan = (prob.tspan[1], min(prob.tspan[2], t[end])),
            p = ps
        )
    end
    prob_func(prob, i, repeat) = sample_prob(prob)
    prob_func(prob, ctx) = sample_prob(prob)
    eprob = EnsembleProblem(prob, prob_func = prob_func)

    indexof(sym, syms) = indexin(Symbol.(sym), Symbol.(syms))
    idx = indexof(sym, ModelingToolkit.unknowns(prob.f.sys))

    esol = solve(
        eprob, nothing, EnsembleSerial(), saveat = t, trajectories = samples,
        save_idxs = idx
    )
    return [
        Array(reduce(hcat, SciMLBase.EnsembleAnalysis.timeseries_point_quantile(esol, q, t).u)')
            for q in quants
    ]
end

"""
    plot_uncertainty_forecast(
            prob, sym, t, uncertainp, samples; label, kwargs...
        ) -> Nothing

Plot sampled trajectories of `sym` under uncertain parameters.

# Arguments

  - `prob`: a SciML problem to simulate.
  - `sym`: symbolic states or observables to plot.
  - `t`: ordered saved times within `prob.tspan`.
  - `uncertainp`: parameter-to-distribution pairs.
  - `samples::Integer`: number of trajectories to plot.

# Keywords

  - `label`: series labels. Defaults to names derived from `sym`.
  - `kwargs...`: keyword arguments forwarded to `Plots.plot` and `Plots.plot!`.

# Returns

  - `nothing` after displaying the plot of all sampled trajectories.

# Examples

```julia
plot_uncertainty_forecast(prob, x, 0.0:0.1:10.0, [k => Uniform(0.8, 1.2)], 25)
```
"""
function plot_uncertainty_forecast(
        prob, sym, t, uncertainp, samples;
        label = reshape(string.(Symbol.(sym)), 1, length(sym)),
        kwargs...
    )
    esol = get_uncertainty_forecast(prob, sym, t, uncertainp, samples)
    p = plot(Array(esol[1]'), idxs = sym; label = label, kwargs...)
    for i in 2:samples
        plot!(p, Array(esol[i]'), idxs = sym; label = false, kwargs...)
    end
    return display(p)
end

"""
    plot_uncertainty_forecast_quantiles(
            prob, sym, t, uncertainp, samples,
            quants = (0.05, 0.95); label = false, kwargs...
        ) -> Plots.Plot

Plot pointwise uncertainty quantiles of `sym`.

# Arguments

  - `prob`: a SciML problem to simulate.
  - `sym`: the symbolic state or observable to plot.
  - `t`: ordered saved times within `prob.tspan`.
  - `uncertainp`: parameter-to-distribution pairs.
  - `samples::Integer`: number of ensemble trajectories.
  - `quants`: quantile probabilities to plot.

# Keywords

  - `label`: label for the first quantile series. Defaults to `false`.
  - `kwargs...`: keyword arguments forwarded to `Plots.plot` and `Plots.plot!`.

# Returns

  - The plot containing one series per requested quantile.

# Examples

```julia
plot_uncertainty_forecast_quantiles(
    prob, x, 0.0:0.1:10.0, [k => Uniform(0.8, 1.2)], 100
)
```
"""
function plot_uncertainty_forecast_quantiles(
        prob, sym, t, uncertainp, samples,
        quants = (0.05, 0.95); label = false,
        kwargs...
    )
    qs = get_uncertainty_forecast_quantiles(prob, sym, t, uncertainp, samples, quants)
    plot(t, qs[1]; label = label, kwargs...)
    return plot!(t, qs[2]; label = false, kwargs...)
end
