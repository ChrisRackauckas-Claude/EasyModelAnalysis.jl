"""
    ensemble_weights(sol::EnsembleSolution, data_ensem)

Fit unconstrained linear weights that combine ensemble trajectories to match data.

For every observed quantity, the trajectory predictions are stacked with the provided
measurements and solved as a least-squares system. The resulting weights are not
constrained to be nonnegative or to sum to one.

## Arguments

  - `sol`: An `EnsembleSolution` containing one prediction trajectory per candidate
    model.
  - `data_ensem`: A collection of `symbol => values` or `symbol => (times, values)`
    pairs. The time entries, when present, are not used; the values must correspond to
    the saved prediction times in `sol`.

## Returns

  - A vector of least-squares weights, with one entry per trajectory in `sol`.

## Examples

```julia
weights = ensemble_weights(sol, [x => observed_x, y => observed_y])
```

!!! note

    This function assumes that the saved times in `sol` match the measurement times in
    `data_ensem`.
"""
function ensemble_weights(sol::EnsembleSolution, data_ensem)
    obs = first.(data_ensem)
    predictions = reduce(
        vcat, reduce(hcat, [sol.u[i][s] for i in 1:length(sol.u)]) for s in obs
    )
    data = reduce(
        vcat,
        [
            data_ensem[i][2] isa Tuple ? data_ensem[i][2][2] : data_ensem[i][2]
                for i in 1:length(data_ensem)
        ]
    )
    return weights = predictions \ data
end

struct EnsembleProbForwarder{P}
    all_probs::P
end

(f::EnsembleProbForwarder)(prob, i::Integer, repeat) = f.all_probs[i]
(f::EnsembleProbForwarder)(prob, ctx) = f.all_probs[ctx.sim_id]

"""
    bayesian_ensemble(
        probs, ps, datas;
        noise_prior = InverseGamma(2, 3),
        mcmcensemble = Turing.MCMCSerial(),
        nchains = 4, niter = 1_000, keep = 100
    )

Construct an `EnsembleProblem` from posterior samples of separately calibrated models.

For each entry, [`bayesian_datafit`](@ref) produces posterior samples. Problems remade
with the tail of those samples are concatenated into one ensemble, whose trajectory
index selects the corresponding calibrated problem. The current index range includes
both endpoints, so `keep` retains `keep + 1` posterior samples per model when enough
samples are available.

## Arguments

  - `probs`: A collection of ODE problems, one for each model to calibrate.
  - `ps`: A collection whose `i`th entry is the symbolic-parameter `=> prior`
    specification passed to [`bayesian_datafit`](@ref) for `probs[i]`.
  - `datas`: A collection whose `i`th entry is data in the format accepted by
    [`bayesian_datafit`](@ref) for `probs[i]`.

## Keyword Arguments

  - `noise_prior`: Prior distribution on the observation noise passed to
    [`bayesian_datafit`](@ref). Defaults to `InverseGamma(2, 3)`.
  - `mcmcensemble`: The Turing MCMC ensemble strategy used for sampling. Defaults to
    `Turing.MCMCSerial()`.
  - `nchains`: Number of MCMC chains per model. Defaults to `4`.
  - `niter`: Number of MCMC iterations per chain. Defaults to `1_000`.
  - `keep`: Tail offset used to collect posterior draws. The implementation includes
    indices `n - keep:n`, so it retains `keep + 1` draws per model when valid. Defaults
    to `100`.

## Returns

  - An `EnsembleProblem` whose trajectories correspond to the collected
    posterior-sample problems. Solve it with a trajectory count matching the retained
    posterior problems, then use [`ensemble_weights`](@ref) to fit model weights.

## Examples

```julia
ensemble_prob = bayesian_ensemble(probs, priors, datasets; niter = 1_000, keep = 100)
```
"""
function bayesian_ensemble(
        probs, ps, datas;
        noise_prior = InverseGamma(2, 3),
        mcmcensemble = Turing.MCMCSerial(),
        nchains = 4,
        niter = 1_000,
        keep = 100
    )
    fits = map(probs, ps, datas) do prob, p, data
        bayesian_datafit(prob, p, data; noise_prior, mcmcensemble, nchains, niter)
    end

    models = map(probs, fits) do prob, fit
        [
            remake(prob, p = Pair.(first.(fit), getindex.(last.(fit), i)))
                for i in (length(fit[1][2]) - keep):length(fit[1][2])
        ]
    end

    @info "Calibrations are complete"

    all_probs = reduce(vcat, models)

    @info "$(length(all_probs)) total models"

    return enprob = EnsembleProblem(
        all_probs[1]; prob_func = EnsembleProbForwarder(all_probs)
    )
end
