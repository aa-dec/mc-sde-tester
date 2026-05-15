using StaticArrays
using Simulator

# Default 2D OU process
A = SA[-1.0f0 0.0f0 ; 0.0f0 -1.0f0]
B = SA[1.0f0 0 ; 0 1.0f0]
p = (A=A, B=B)

function drift(u, p, t)
    du1 = p.A[1,1]*u[1] + p.A[1,2]*u[2]
    du2 = p.A[2,1]*u[1] + p.A[2,2]*u[2]
    
    return SA[du1, du2]
end
function noise(u, p, t)
    # B is 2x2, assuming additive diagonal noise for this OU process
    # Adjust indexing if B is intended to be a vector or non-diagonal
    s2 = sqrt(2.0f0)
    du1 = s2 * p.B[1,1]
    du2 = s2 * p.B[2,2]
    return SA[du1, du2]
end

n_traj = 100
T_end = 100.0f0
burnin_time = 5.0f0
dt::Float32     = 0.1f0
dim = 2
fname ="linear-ou.dat"

experiment = SDEExperiment(
    drift,
    noise,
    p; 
    n_traj=n_traj, T_end=T_end, dt=dt, dim=dim, fname=fname
)
inits = burnin(experiment, burnin_time)
run_experiment(experiment, inits)

