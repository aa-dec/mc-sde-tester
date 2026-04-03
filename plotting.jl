using JLD2
using Plots
using StatsPlots

using Simulator

fname = "linear-ou"

cwd = pwd()

data = load(joinpath(cwd, "data", fname * ".dat"))
trajectories = data["cpu_data"]
timesteps = data["timesteps"]
n_traj = data["sde_exp"].n_traj
T_end = data["sde_exp"].T_end
dt = data["sde_exp"].dt

# plotting histogram
xs = [traj[1,:] for traj in trajectories]
ys = [traj[2,:] for traj in trajectories]

x = reduce(vcat, xs)
y = reduce(vcat, ys)
n_points = length(x)
histogram2d(x, y, bins=100, title="Histogram of 2D linear OU: $(n_points) points")
savefig(joinpath(cwd, "plot", fname*"-hist.svg"))

# sample trajectory
end_time = 10
end_idx = Int(10/dt)
plot(xs[1][1:end_idx], ys[1][1:end_idx], 
     title = "Example Trajectory", 
     xlabel = "x", 
     ylabel = "y", 
     lw = 2)
savefig(joinpath(cwd, "plot", fname*"-traj.svg"))