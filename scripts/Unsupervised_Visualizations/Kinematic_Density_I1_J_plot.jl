# ==============================================================================
# Visualization: 2D Kinematic Density in Invariant Space (I₁ vs J)
# ------------------------------------------------------------------------------
# Generates a hexbin density plot of all kinematic states (I₁, J) observed across
# all elements and load steps of the 3D perforated plate FE simulation.
#
# Also plots the physical admissibility boundary I₁ >= 3 * J^(2/3) (the theoretical
# limit where all three principal stretches are real and positive).
# ==============================================================================

using DrWatson
@quickactivate "Sparse_PGD_NN"
using JLD2, CairoMakie, Statistics, Random

set_theme!(theme_latexfonts(), fontsize=20, linewidth=3)

# Load full-field kinematic identification data
data_file = datadir("sims", "euclid_identification_data.jld2")
if !isfile(data_file)
    error("Data file not found at $data_file! Run scripts/Unsupervised/Hyperelasticity_Data_Generation.jl first.")
end
d = load(data_file)
I1_hist = d["I1_history"]
J_hist = d["J_history"]
n_steps = length(I1_hist)
n_cells = length(I1_hist[1])

# Extract all points across all load steps
all_I1 = Float32[]
all_J = Float32[]
sizehint!(all_I1, n_steps * n_cells)
sizehint!(all_J, n_steps * n_cells)

for s in 1:n_steps
    append!(all_I1, I1_hist[s])
    append!(all_J, J_hist[s])
end

# Create single-panel density figure
fig = Figure(size=(800, 650))

# Continuum mechanics physical admissibility boundary: I1 >= 3 * J^(2/3)
j_curve = range(0.7f0, 1.9f0, length=200)
i1_boundary = 3.0f0 .* (j_curve .^ (2 / 3f0))

# -------------------------------------------------------------
# Hexbin 2D Kinematic Density (Log scale)
# -------------------------------------------------------------
ax1 = Axis(fig[1, 1],
    #title=L"\text{Kinematic Density in Invariant Space: }\log_{10}(\text{Count})",
    xlabel=L"I_1",
    ylabel=L"J",
    xgridvisible=true,
    ygridvisible=true)

hb = hexbin!(ax1, all_I1, all_J, bins=80, colorscale=log10, colormap=:viridis)
lines!(ax1, i1_boundary, j_curve, color=:red, linestyle=:dash, linewidth=2.5, label=L"I_1 = 3 J^{2/3}\ \text{(Limit)}")
Colorbar(fig[1, 2], hb, label=L"\log_{10}(\text{Point Count})", width=18)
axislegend(ax1, position=:rb, framevisible=true)

# Set axis limits
xlims!(ax1, 2.5, 15.0)
ylims!(ax1, 0.75, 1.85)

out_file = projectdir("plots/EUCLID/Kinematic_Density_I1_J.png")
mkpath(dirname(out_file))
save(out_file, fig)
println("Saved figure to: ", out_file)
