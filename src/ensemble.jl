"""
    ensemble_weights(sol::EnsembleSolution, data_ensem) -> AbstractVector

Returns the weights for a linear combination of the models
so that the prediction = sum(weight[i] * model_prediction[i])
where `sol` is the ensemble solution and `data_ensem` is the
dataset on which the ensembler should be trained on.

# Arguments

  - `sol`: ensemble solution whose trajectories provide model predictions.
  - `data_ensem`: pairs from symbolic states to measurements used to fit the combination.

# Returns

  - Least-squares weights for the linear combination of ensemble predictions.

!!! note

    This function currently assumes that `sol.t` matches the time points of all measurements
    in `data_ensem`.

# Examples

```julia
weights = ensemble_weights(ensemble_solution, [x => observations])
```
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

"""
    EnsembleProbForwarder(all_probs)

Callable used as the `prob_func` of the `EnsembleProblem` returned by
[`bayesian_ensemble`](@ref). It selects the per-trajectory problem from the stored
`all_probs` vector. It supports both the `prob_func(prob, ctx)` interface of newer
SciMLBase (selecting via `ctx.sim_id`) and the legacy `prob_func(prob, i, repeat)`
interface (selecting via the integer index). Storing `all_probs` lets callers recover
the number of trajectories via `enprob.prob_func.all_probs`.
"""
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

Build an ensemble of calibrated models by Bayesian-fitting each model to its own data
and collecting posterior samples of the fitted problems into a single `EnsembleProblem`.

For each model, [`bayesian_datafit`](@ref) is run to obtain posterior samples of the
parameters, and the last `keep` posterior draws are turned into `remake`d problems (one
problem per draw). The problems from all models are concatenated, and the returned
`EnsembleProblem` uses an internal `prob_func` that selects the `i`th collected problem so
that the `i`th trajectory solves it. Solving the returned problem with
`trajectories = length(enprob.prob_func.all_probs)` therefore samples the full posterior
ensemble across all models.

# Arguments

  - `probs`: a vector of `ODEProblem`s, one per model to be calibrated.
  - `ps`: a vector where the `i`th entry is the parameter specification (a vector of
    symbolic-parameter `=> prior` pairs) passed to [`bayesian_datafit`](@ref) for
    `probs[i]`.
  - `datas`: a vector where the `i`th entry is the data (of the form accepted by
    [`bayesian_datafit`](@ref)) used to calibrate `probs[i]`.

# Keywords

  - `noise_prior`: prior distribution on the observation noise passed to
    [`bayesian_datafit`](@ref). Defaults to `InverseGamma(2, 3)`.
  - `mcmcensemble`: the Turing MCMC ensemble strategy used for sampling. Defaults to
    `Turing.MCMCSerial()`.
  - `nchains`: number of MCMC chains per model. Defaults to `4`.
  - `niter`: number of MCMC iterations per chain. Defaults to `1_000`.
  - `keep`: number of posterior draws (from the tail of the chain) kept per model to form
    the ensemble. Defaults to `100`.

# Returns

  - An `EnsembleProblem` whose trajectories correspond to the collected posterior-sample
    problems from all models. Use the resulting ensemble solution together with
    [`ensemble_weights`](@ref) to weight the models against data.

# Examples

```julia
ensemble_problem = bayesian_ensemble([prob1, prob2], priors, datasets; keep = 50)
ensemble_solution = solve(ensemble_problem, Tsit5(); trajectories = 100)
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
