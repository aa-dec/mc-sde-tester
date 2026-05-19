module Simulator

using DiffEqGPU, CUDA, StochasticDiffEq, StaticArrays
using JLD2, Printf, Dates

include("models/Linear.jl")
include("models/Rotate.jl")

using .LinearSDE
using .RotateSDE

export SDEExperiment, run_experiment, linear_drift, linear_noise, burnin

Base.@kwdef struct SDEExperiment{D, N, P}
    drift::D 
    noise::N 
    params::P 
    n_traj::Int     = 100
    T_end::Float32  = 1000.0f0
    dt::Float32     = 0.01f0
    dim::Int
    seed::Int     =1234
    fname::Union{String, Nothing}   =nothing
end

#pretty printing
function Base.show(io::IO, ::MIME"text/plain", ex::SDEExperiment)
    println(io, "──────────────────────────────────────────────────────────")
    println(io, "SDE Markov Simulation")
    println(io, "──────────────────────────────────────────────────────────")
    @printf(io, "  %-15s : %dD\n", "Dimensions", ex.dim)
    @printf(io, "  %-15s : %d\n", "Trajectories", ex.n_traj)
    @printf(io, "  %-15s : %.2f (dt: %.3f)\n", "Time End", ex.T_end, ex.dt)
    @printf(io, "  %-15s : %d\n", "Seed", ex.seed)
    println(io, "──────────────────────────────────────────────────────────")
    println(io, "  Parameters Type: ", typeof(ex.params))
    println(io, "  Drift Function:  ", ex.drift)
    println(io, "  Noise Function:  ", ex.noise)
    println(io, "──────────────────────────────────────────────────────────")
end

# constructor 
function SDEExperiment(drift, noise, params; dim, kwargs...)
    if !isbits(params)
        throw(ArgumentError("Params must be isbits for GPU compatibility. Received: $(typeof(params))"))
    end
    return SDEExperiment(; drift=drift, noise=noise, params=params, dim=dim, kwargs...)
end

function burnin(sde_exp::SDEExperiment, burnin_time)
    display(sde_exp)
    CUDA.allowscalar(false)

    if CUDA.functional()
        @info "NVIDIA GPU detected. Running simulations via CUDA acceleration."
        ensemble_backend = EnsembleGPUArray(CUDA.CUDABackend())
    else
        @warn "No NVIDIA GPU found. Falling back to multi-threaded CPU parallelization."
        ensemble_backend = EnsembleThreads()
    end
    
    # burnin
    @info "Starting Burn-in Phase" details="""
    Gathering an ensemble of initial conditions 
    distributed according to the invariant measure.
    """ burnin_time=burnin_time
    

    u0 = rand(SVector{sde_exp.dim, Float32})
    prob = SDEProblem(sde_exp.drift, sde_exp.noise, u0, (0.0f0, burnin_time), sde_exp.params)
    ensemble_prob = EnsembleProblem(prob, 
        prob_func = (prob, ctx) -> 
            remake(prob; u0 = rand(SVector{sde_exp.dim, Float32})),
            output_func = (sol, ctx) -> (sol.u[end], false) 
    )

    time_0 = time()
    sol = solve(ensemble_prob, SOSRI(), ensemble_backend, 
        trajectories = sde_exp.n_traj, adaptive = false)
    # Initial conditions roughly distributed according to the invariant measures
    time_elapsed = time() - time_0 
    @info "Time taken on burnin: " time=time_elapsed
    cpu_data = sol.u

    return cpu_data
end

function run_experiment(sde_exp::SDEExperiment, inits)

    display(sde_exp)
    CUDA.allowscalar(false)

    # Generate trajectories
    @info "Generating Data" details="""
    Simulating an ensemble of trajectories.
    """ end_time=sde_exp.T_end dt=sde_exp.dt

    if CUDA.functional()
        @info "NVIDIA GPU detected. Running simulations via CUDA acceleration."
        ensemble_backend = EnsembleGPUArray(CUDA.CUDABackend())
    else
        @warn "No NVIDIA GPU found. Falling back to multi-threaded CPU parallelization."
        ensemble_backend = EnsembleThreads()
    end

    prob = SDEProblem(sde_exp.drift, sde_exp.noise, inits[1], (0.0f0, sde_exp.T_end), sde_exp.params)
    prob_func = (prob, ctx) -> remake(prob, u0 = inits[ctx.sim_id])
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    
    time_0 = time()
    sol = solve(ensemble_prob, SOSRI(), ensemble_backend, 
        trajectories = sde_exp.n_traj, saveat = sde_exp.dt, adaptive = false)
    time_elapsed = time() - time_0 
    @info "Time taken on simulation: " time=time_elapsed
    
    date_time = now()
    save_dir = joinpath(pwd(), "data")
    mkpath(save_dir)
    cpu_data = sol.u 

    if !isnothing(sde_exp.fname)
        filename=joinpath(save_dir, sde_exp.fname)

        @info "Saving data" details="""
        Date = $(date_time)
        Filename = $(filename)
        """ date_time=date_time filename=sde_exp.fname
        timesteps = collect(0.0f0:sde_exp.dt:sde_exp.T_end)
        jldsave(filename; sde_exp, cpu_data, timesteps)
    end

    return cpu_data
end


end;