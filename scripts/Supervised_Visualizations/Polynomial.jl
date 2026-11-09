using DrWatson
@quickactivate "Sparse_PGD_NN"

using Flux
using CairoMakie
using Statistics
using Random
using LinearAlgebra
using Dates

Random.seed!(42)
Makie.inline!(true)

set_theme!(theme_latexfonts(), fontsize=30, linewidth=4)

###############################################################################################################
# 1. Reference Strain Energy Function and Analytical Derivatives
###############################################################################################################
const C10 = 0.5f0
const C02 = 0.2f0

"""
Reference strain energy function:
Ψ(I₁, I₂) = C₁₀ * (I₁ - 3)² + C₀₂ * (I₂ - 3)
"""
function reference_function(x)
    return C10 * (x[1] - 3f0)^2 + C02 * (x[2] - 3f0)
end

"""
Analytical gradient of reference energy with respect to x = [I1, I2]
"""
function grad_psi_ref(x)
    I1, I2 = x[1], x[2]
    d_I1 = 2.0f0 * C10 * (I1 - 3f0)
    d_I2 = C02
    return Float32[d_I1, d_I2]
end

###############################################################################################################
# 2. Load Trained Neural Network from datadir("exp_raw")
###############################################################################################################
function load_latest_network(prefix="Polynomial")
    # Check candidate checkpoint directories
    cand_paths = [
        datadir("exp_raw", "Final", "E1_incompressible_RegHigh", "network.jld2"),
        datadir("exp_raw", "Final", "E1_incompressible", "network.jld2"),
    ]
    for c in cand_paths
        if isfile(c)
            println("Loading network from: $c")
            return wload(c), basename(dirname(c))
        end
    end

    exp_raw_dir = datadir("exp_raw")
    if isdir(exp_raw_dir)
        subdirs = filter(d -> isdir(joinpath(exp_raw_dir, d)) && isfile(joinpath(exp_raw_dir, d, "network.jld2")), readdir(exp_raw_dir))
        poly_dirs = filter(d -> occursin("Polynomial", d) || occursin("incompressible", d) || occursin("E1", d) || occursin("D1", d), subdirs)
        if !isempty(poly_dirs)
            sort!(poly_dirs)
            latest_dir = joinpath(exp_raw_dir, poly_dirs[end])
            println("Loading trained model from: $latest_dir")
            data = wload(joinpath(latest_dir, "network.jld2"))
            return data, basename(latest_dir)
        end
    end
    error("No trained polynomial network found in $(datadir("exp_raw")). Please run training first (scripts/Supervised/Polynomial.jl).")
end

model_data, folder_name = load_latest_network()
run_name = folder_name
nets_x1 = model_data["nets_x1"]
nets_x2 = model_data["nets_x2"]
raw_scale = model_data["raw_scale"]
num_modes = model_data["num_modes"]

get_scale(scale_param, m=1) = length(scale_param) == 1 ? (softplus(scale_param[1]) + 1.0f0) : (softplus(scale_param[m]) + 1.0f0)

"""
Neural Network Strain Energy Potential Ψ_NN(x) for x = [I1, I2] (incompressible)
"""
function psi_nn(x)
    total = zero(eltype(x))
    for m in 1:num_modes
        S = get_scale(raw_scale, m)
        val = nets_x1[m](x[1:1])[1] + nets_x2[m](x[2:2])[1]
        total += S * (exp(val / S) - 1.0f0)
    end
    return total
end

"""
Gradient of NN energy function with respect to x = [I1, I2] via automatic differentiation
"""
function grad_psi_nn(x)
    return Flux.gradient(psi_nn, Float32.(x))[1]
end

###############################################################################################################
# 3. Continuum Mechanics: Incompressible Hyperelasticity (Option A: J = 1)
###############################################################################################################
"""
Compute the Second Piola-Kirchhoff stress tensor S and First Piola-Kirchhoff (nominal) stress P
for an incompressible material (J = 1) with strain energy potential Ψ(I1, I2).

For incompressible materials:
  S = -p * C^(-1) + 2*(∂Ψ/∂I1)*I + 2*(∂Ψ/∂I2)*(I1*I - C)
  P = F * S

The hydrostatic pressure p is determined as a Lagrange multiplier by enforcing zero normal stress
along the unconstrained (traction-free) direction k (i.e., S[k, k] = 0):
  p = [ 2*(∂Ψ/∂I1) + 2*(∂Ψ/∂I2)*(I1 - C[k, k]) ] / (C^(-1))[k, k]
"""
function compute_incompressible_stress(F, grad_fn; free_dim=2)
    C = F' * F
    I1 = tr(C)
    invC = inv(C)

    # Invariants for incompressible state (J = 1)
    I2 = 0.5f0 * (I1^2 - tr(C * C))
    x = Float32[I1, I2]

    dPsi = grad_fn(x)
    dPsi_dI1 = dPsi[1]
    dPsi_dI2 = dPsi[2]

    I3x3 = Matrix{Float32}(I, 3, 3)

    # Lagrange multiplier p enforcing S[free_dim, free_dim] = 0
    p = (2.0f0 * dPsi_dI1 + 2.0f0 * dPsi_dI2 * (I1 - C[free_dim, free_dim])) / invC[free_dim, free_dim]

    S = -p .* invC .+ 2.0f0 * dPsi_dI1 .* I3x3 .+ 2.0f0 * dPsi_dI2 .* (I1 .* I3x3 .- C)
    P = F * S
    sigma = F * S * F'  # J = 1

    return (S=S, P=P, sigma=sigma, p=p)
end

###############################################################################################################
# 4. Homogeneous Incompressible Deformation Modes (Kinematically Exact)
###############################################################################################################
# 1. Uniaxial Tension / Compression: λ1 = λ, λ2 = λ3 = 1/√λ, free_dim = 2
f_ut_F(lam) = Float32[lam 0 0; 0 1.0f0/sqrt(lam) 0; 0 0 1.0f0/sqrt(lam)]

# 2. Pure Shear / Planar Tension: λ1 = λ, λ2 = 1, λ3 = 1/λ, free_dim = 3
f_ps_F(lam) = Float32[lam 0 0; 0 1.0f0 0; 0 0 1.0f0/lam]

# 3. Equi-biaxial Tension / Compression: λ1 = λ, λ2 = λ, λ3 = 1/λ^2, free_dim = 3
f_eb_F(lam) = Float32[lam 0 0; 0 lam 0; 0 0 1.0f0/(lam^2)]

"""
Evaluates P_11 directly across stretch_range for a given incompressible mode.
No numerical bisection or root-finding is required.
"""
function evaluate_incompressible_mode(stretch_range, f_F, free_dim, grad_fn)
    P11_vals = zeros(Float32, length(stretch_range))
    for (i, lam) in enumerate(stretch_range)
        F = f_F(lam)
        res = compute_incompressible_stress(F, grad_fn; free_dim=free_dim)
        P11_vals[i] = res.P[1, 1]
    end
    return P11_vals
end

###############################################################################################################
# 5. Evaluate Loading Curves over Stretch Range
###############################################################################################################
stretch_range = range(0.6f0, 3.0f0, length=120)
stretch_range_biaxial = range(0.6f0, 2.5f0, length=120)

println("\nEvaluating incompressible deformation modes (Uniaxial, Pure Shear, Equi-biaxial)...")
P11_ut_ref = evaluate_incompressible_mode(stretch_range, f_ut_F, 2, grad_psi_ref)
P11_ut_nn = evaluate_incompressible_mode(stretch_range, f_ut_F, 2, grad_psi_nn)

P11_ps_ref = evaluate_incompressible_mode(stretch_range, f_ps_F, 3, grad_psi_ref)
P11_ps_nn = evaluate_incompressible_mode(stretch_range, f_ps_F, 3, grad_psi_nn)

P11_eb_ref = evaluate_incompressible_mode(stretch_range_biaxial, f_eb_F, 3, grad_psi_ref)
P11_eb_nn = evaluate_incompressible_mode(stretch_range_biaxial, f_eb_F, 3, grad_psi_nn)
println("Evaluation complete.")

###############################################################################################################
# 6. Single [1, 1] Plot: P_11 vs Stretch across All Loading Modes
###############################################################################################################
fig = Figure(size=(900, 650))

ax = Axis(fig[1, 1],
    #title="Nominal Stress P₁₁ vs. Stretch λ (Incompressible Polynomial / Mooney-Rivlin)\nReference vs. Neural Network Model ($run_name)",
    xlabel="Stretch λ [-]",
    ylabel="Nominal Stress P₁₁ [MPa]",
    xgridvisible=true,
    ygridvisible=true
)

# Reference curves (Solid lines)
lines!(ax, stretch_range, P11_ut_ref, color=:crimson, linewidth=2.5, label="Uniaxial (Analytical)")
lines!(ax, stretch_range, P11_ps_ref, color=:royalblue, linewidth=2.5, label="Pure Shear (Analytical)")
lines!(ax, stretch_range_biaxial, P11_eb_ref, color=:forestgreen, linewidth=2.5, label="Equi-biaxial (Analytical)")

# Neural Network curves (Dashed lines)
scatter!(ax, stretch_range[1:5:end], P11_ut_nn[1:5:end], color=:crimson, marker='o', markersize=20, label="Uniaxial (Network)")
scatter!(ax, stretch_range[1:5:end], P11_ps_nn[1:5:end], color=:royalblue, marker='o', markersize=20, label="Pure Shear (Network)")
scatter!(ax, stretch_range_biaxial[1:5:end], P11_eb_nn[1:5:end], color=:forestgreen, marker='o', markersize=20, label="Equi-biaxial (Network)")

# Reference lines at undeformed state (λ = 1, P = 0)
vlines!(ax, [1.0f0], color=:gray, linestyle=:dot, linewidth=1.5)
hlines!(ax, [0.0f0], color=:gray, linestyle=:dot, linewidth=1.5)

axislegend(ax, position=:rb, framevisible=true, backgroundcolor=(:white, 0.9))

save_dir_plots = plotsdir("Visualizations", folder_name)
mkpath(save_dir_plots)
save_plot_path = joinpath(save_dir_plots, "Polynomial_Loading_Modes.png")
save(save_plot_path, fig)
printstyled("\nFigure saved successfully to: $save_plot_path\n", color=:green, bold=true)

fig