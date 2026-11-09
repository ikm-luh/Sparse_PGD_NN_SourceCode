using DrWatson
@quickactivate "Sparse_PGD_NN"

using Flux
using SparseArrays
using LinearAlgebra
using Statistics
using Random
using Dates
using JLD2
using Optim
using CairoMakie

# Set random seed for reproducibility
Random.seed!(42)
Makie.inline!(true)

println("="^75)
println("EUCLID: Unsupervised Constitutive Law Identification via Sparse PGD-NN")
println("Noise Robustness Benchmark (Simulating Experimental DIC Strain Measurement Noise)")
println("Physics-Informed Discovery using Ferrite.jl, Flux.jl, and Optim.jl (L-BFGS)")
println("="^75)

# ----------------------------------------------------------------------
# 1. Load Experimental Data (Kinematics & Reaction Forces from FE)
# ----------------------------------------------------------------------
data_path = datadir("sims", "euclid_identification_data.jld2")
if !isfile(data_path)
    error("Data file not found at $data_path! Run Hyperelasticity_Ferrite.jl first.")
end

println("\n[1/6] Loading full-field dataset from: ", data_path)
data = wload(data_path)

n_cells = data["n_cells"]
ndofs = data["ndofs"]
n_steps_total = data["n_steps"]
free_dofs = data["free_dofs"]
pull_dofs_x = data["pull_dofs_x"]
F_ext_history = data["F_ext_history"]
K1_history = data["K1_history"]
K2_history = data["K2_history"]
K3_history = data["K3_history"]
I1_history = data["I1_history"]
I2_history = data["I2_history"]
J_history = data["J_history"]
mat_params = data["material_params"]

μ_true = mat_params["μ"]
λ_true = mat_params["λ"]

# Select evenly spaced steps across the total load history (spanning small, medium, and high stretch)
const step_indices = [25, 50, 75, 100]
const n_steps = length(step_indices)

println("  • Elements (cells):        $n_cells")
println("  • Total DOFs:              $ndofs")
println("  • Free DOFs:               $(length(free_dofs))")
println("  • Pull DOFs (x):           $(length(pull_dofs_x))")
println("  • Total Load Steps in FE:  $n_steps_total")
println("  • Selected Training Steps: $step_indices")
println("  • Experimental F_ext ($n_steps):  $(round.(F_ext_history[step_indices], digits=2)) N")
println("  • Ground-Truth Parameters: μ = $μ_true, λ = $λ_true")

# Preconvert FEM virtual work operators and clean invariants to Float32 for optimal AD performance ($n_steps selected steps)
const K1_f32 = [SparseMatrixCSC{Float32,Int}(K1_history[s]) for s in step_indices]
const K2_f32 = [SparseMatrixCSC{Float32,Int}(K2_history[s]) for s in step_indices]
const K3_f32 = [SparseMatrixCSC{Float32,Int}(K3_history[s]) for s in step_indices]
const F_ext_f32 = Float32.([F_ext_history[s] for s in step_indices])
const I1_clean = [Float32.(I1_history[s]) for s in step_indices]
const I2_clean = [Float32.(I2_history[s]) for s in step_indices]
const J_clean = [Float32.(J_history[s]) for s in step_indices]

# ----------------------------------------------------------------------
# 1b. Add Experimental DIC Strain Measurement Noise (Constant / Static)
# ----------------------------------------------------------------------
# In DIC, camera noise perturbs the displacement field and propagates into
# the strain invariants. Noise is generated ONCE with a fixed seed so it
# remains strictly constant across all Adam and L-BFGS training iterations.

"""
    add_dic_strain_noise(I1_list, I2_list, J_list, lvl; seed=42)

Perturb strain invariants relative to the undeformed reference state (I1=3, I2=3, J=1)
using zero-mean Gaussian noise, mimicking full-field optical DIC measurement errors.
Lower bounds enforce physical admissibility (J > 0, I1 > 0).
"""
function add_dic_strain_noise(I1_list, I2_list, J_list, lvl; seed=42)
    if lvl <= 0.0f0
        println("  • Noise level = 0%: Using clean full-field strain invariants.")
        return I1_list, I2_list, J_list
    end
    Random.seed!(seed)

    I1_noisy = Vector{Vector{Float32}}()
    I2_noisy = Vector{Vector{Float32}}()
    J_noisy = Vector{Vector{Float32}}()

    for s in 1:length(I1_list)
        # Deviation from undeformed reference configuration
        ΔI1 = I1_list[s] .- 3.0f0
        ΔI2 = I2_list[s] .- 3.0f0
        ΔJ = J_list[s] .- 1.0f0

        # Gaussian measurement noise perturbation proportional to local deformation
        n1 = lvl .* randn(Float32, length(ΔI1))
        n2 = lvl .* randn(Float32, length(ΔI2))
        n3 = lvl .* randn(Float32, length(ΔJ))

        # Reconstructed invariants with physical positivity guards
        push!(I1_noisy, max.(1.0f-3, 3.0f0 .+ ΔI1 .* (1.0f0 .+ n1)))
        push!(I2_noisy, max.(1.0f-3, 3.0f0 .+ ΔI2 .* (1.0f0 .+ n2)))
        push!(J_noisy, max.(1.0f-3, 1.0f0 .+ ΔJ .* (1.0f0 .+ n3)))
    end

    println("  • Applied $(round(lvl * 100, digits=1))% constant DIC strain measurement noise to (I₁, I₂, J).")
    return I1_noisy, I2_noisy, J_noisy
end

# Ground truth Neo-Hookean strain energy for validation
Ψ_true(I1, I2, J) = (μ_true / 2.0) * (I1 - 3.0 - 2.0 * log(max(1e-6, J))) + (λ_true / 2.0) * (J - 1.0)^2

const X_REF = Float32[3.0; 3.0; 1.0;;]
const REF_X1 = Float32[3.0;;]
const REF_X2 = Float32[3.0;;]
const REF_X3 = Float32[1.0;;]
split_input(x) = (x[1:1, :], x[2:2, :], x[3:3, :])
inv_sp(w) = @. ifelse(w > 20.0f0, w, log(max(1f-6, exp(w) - 1.0f0)))
get_scale(scale_param, m=1) = length(scale_param) == 1 ? (softplus(scale_param[1]) + 1.0f0) : (softplus(scale_param[m]) + 1.0f0)

# ----------------------------------------------------------------------
# 2. Sparse PGD-NN Architecture (Matching D3 Script)
# ----------------------------------------------------------------------
println("\n[2/6] Initializing Sparse PGD-NN Multi-Mode Architecture...")

"""
    init_model(num_modes)

1D Input Convex Neural Network (ICNN) sub-network template:
  - Layer 1: Dense(1 => 10, softplus) - affine input + convex activation
  - Layer 2: Dense(10 => 1) - non-negative weights W2 >= 0 enforce convexity
  - Layer 3: Dense(1 => 1, bias=false) - non-negative scaling factor W3 >= 0 (with L1 proximal sparsity)
"""
function init_model(num_modes)
    net_template() = Chain(
        Dense(1 => 10, softplus),
        Dense(10 => 1),
        Dense(1 => 1, bias=false)
    )

    nets_x1 = [net_template() for _ in 1:num_modes]
    nets_x2 = [net_template() for _ in 1:num_modes]
    nets_x3 = [net_template() for _ in 1:num_modes]

    # Initialize Layer 2 and Layer 3 weights to be non-negative
    for i in 1:num_modes
        for net in (nets_x1[i], nets_x2[i], nets_x3[i])
            net.layers[2].weight .= abs.(net.layers[2].weight) .* 0.1f0
            net.layers[3].weight .= abs.(net.layers[3].weight) .* 0.01f0
        end
        # Break initial symmetry: assign distinct coordinate affinity to modes
        if i == 1
            nets_x1[i].layers[3].weight .= 0.05f0
            nets_x3[i].layers[3].weight .= 0.05f0
        elseif i == 2
            nets_x2[i].layers[3].weight .= 0.05f0
            nets_x3[i].layers[3].weight .= 0.05f0
        elseif i == 3
            nets_x1[i].layers[3].weight .= 0.05f0
        elseif i == 4
            nets_x2[i].layers[3].weight .= 0.05f0
        end
    end

    # Mode 5 dedicated exclusively to the volumetric x3 contribution:
    if num_modes >= 5
        nets_x1[5].layers[3].weight .= 0.0f0
        nets_x2[5].layers[3].weight .= 0.0f0

        # Bilateral U-ICNN initialization around J = 1.0:
        w_u = Float32[1, 2, 4, 8, 12, -1, -2, -4, -8, -12]
        nets_x3[5].layers[1].weight .= reshape(w_u, 10, 1)
        nets_x3[5].layers[1].bias .= -w_u .* 1.0f0
        nets_x3[5].layers[2].weight .= 0.1f0
        nets_x3[5].layers[2].bias .= 0.0f0
        nets_x3[5].layers[3].weight .= 0.05f0
    end

    models = [Chain(split_input, Parallel(+, nets_x1[i], nets_x2[i], nets_x3[i])) for i in 1:num_modes]
    return nets_x1, nets_x2, nets_x3, models
end

"""
Analytic derivative df/dx for a 1D subnetwork:
f(x) = W3 * (W2 * softplus(W1 * x + b1) + b2)
=> df/dx = W3 * W2 * (σ(W1 * x + b1) .* W1)
"""
function df_dx_subnet(net, x::AbstractMatrix{Float32})
    W1 = net.layers[1].weight # (10, 1)
    b1 = net.layers[1].bias   # (10,)
    W2 = net.layers[2].weight # (1, 10)
    W3 = net.layers[3].weight[1] # scalar
    z1 = W1 * x .+ b1
    sig = σ.(z1)
    # W2 * (sig .* W1) has shape (1, N)
    return W3 .* (W2 * (sig .* W1))
end

"""
Analytic derivative df/dx in reparameterized V-space:
W2 = softplus(V2), W3 = softplus(V3)
"""
function df_dx_subnet_reparam(net_v, x::AbstractMatrix{Float32})
    W1 = net_v.layers[1].weight
    b1 = net_v.layers[1].bias
    W2 = softplus.(net_v.layers[2].weight)
    W3 = softplus(net_v.layers[3].weight[1])
    z1 = W1 * x .+ b1
    sig = σ.(z1)
    return W3 .* (W2 * (sig .* W1))
end

"""
Compute reference hydrostatic pressure p0 at undeformed state X_REF = [3.0, 3.0, 1.0]:
p0 = 2(∂Ψ/∂I₁) + 4(∂Ψ/∂I₂) + ∂Ψ/∂J
Subtracting p0 from ∂Ψ/∂J guarantees a stress-free reference configuration (P(I) = 0).
"""
function compute_p0(models, raw_scale)
    d1_0 = sum(
        let S = get_scale(raw_scale, m)
            exp.(models[m](X_REF) ./ S) .* df_dx_subnet(models[m].layers[2][1], REF_X1)
        end for m in 1:length(models)
    )
    d2_0 = sum(
        let S = get_scale(raw_scale, m)
            exp.(models[m](X_REF) ./ S) .* df_dx_subnet(models[m].layers[2][2], REF_X2)
        end for m in 1:length(models)
    )
    dJ_0 = sum(
        let S = get_scale(raw_scale, m)
            exp.(models[m](X_REF) ./ S) .* df_dx_subnet(models[m].layers[2][3], REF_X3)
        end for m in 1:length(models)
    )
    return 2.0f0 * d1_0[1] + 4.0f0 * d2_0[1] + dJ_0[1]
end

function compute_p0_reparam(models_v, raw_scale)
    d1_0 = sum(
        let S = get_scale(raw_scale, m),
            out_ref = eval_model_reparam(models_v[m], X_REF)

            exp.(out_ref ./ S) .* df_dx_subnet_reparam(models_v[m].layers[2][1], REF_X1)
        end for m in 1:length(models_v)
    )
    d2_0 = sum(
        let S = get_scale(raw_scale, m),
            out_ref = eval_model_reparam(models_v[m], X_REF)

            exp.(out_ref ./ S) .* df_dx_subnet_reparam(models_v[m].layers[2][2], REF_X2)
        end for m in 1:length(models_v)
    )
    dJ_0 = sum(
        let S = get_scale(raw_scale, m),
            out_ref = eval_model_reparam(models_v[m], X_REF)

            exp.(out_ref ./ S) .* df_dx_subnet_reparam(models_v[m].layers[2][3], REF_X3)
        end for m in 1:length(models_v)
    )
    return 2.0f0 * d1_0[1] + 4.0f0 * d2_0[1] + dJ_0[1]
end

"""
Compute dPsi/dI1, dPsi/dI2, dPsi/dJ across all modes analytically with stress-free reference correction:
Psi = sum_m S_m * ( exp(Phi_m(x) / S_m) - exp(Phi_m(X_REF) / S_m) ) - p0 * (J - 1)
dPsi/dx_k = sum_m exp(Phi_m(x) / S_m) * (dPhi_m/dx_k)
dPsi/dJ = sum_m exp(Phi_m(x) / S_m) * (dPhi_m/dJ) - p0
"""
function eval_derivatives_all(models, raw_scale, I1::Vector{Float32}, I2::Vector{Float32}, J::Vector{Float32})
    N = length(I1)
    x1 = reshape(I1, 1, N)
    x2 = reshape(I2, 1, N)
    x3 = reshape(J, 1, N)
    X = [x1; x2; x3]

    dPsi_dI1 = sum(
        let S = get_scale(raw_scale, m)
            exp_factor = exp.(models[m](X) ./ S)
            exp_factor .* df_dx_subnet(models[m].layers[2][1], x1)
        end for m in 1:length(models)
    )

    dPsi_dI2 = sum(
        let S = get_scale(raw_scale, m)
            exp_factor = exp.(models[m](X) ./ S)
            exp_factor .* df_dx_subnet(models[m].layers[2][2], x2)
        end for m in 1:length(models)
    )

    dPsi_dJ = sum(
        let S = get_scale(raw_scale, m)
            exp_factor = exp.(models[m](X) ./ S)
            exp_factor .* df_dx_subnet(models[m].layers[2][3], x3)
        end for m in 1:length(models)
    )

    p0 = compute_p0(models, raw_scale)
    return vec(dPsi_dI1), vec(dPsi_dI2), vec(dPsi_dJ .- p0)
end

function eval_derivatives_all_reparam(models_v, raw_scale, I1::Vector{Float32}, I2::Vector{Float32}, J::Vector{Float32})
    N = length(I1)
    x1 = reshape(I1, 1, N)
    x2 = reshape(I2, 1, N)
    x3 = reshape(J, 1, N)
    X = [x1; x2; x3]

    dPsi_dI1 = sum(
        let S = get_scale(raw_scale, m),
            out = eval_model_reparam(models_v[m], X),
            exp_factor = exp.(out ./ S)

            exp_factor .* df_dx_subnet_reparam(models_v[m].layers[2][1], x1)
        end for m in 1:length(models_v)
    )

    dPsi_dI2 = sum(
        let S = get_scale(raw_scale, m),
            out = eval_model_reparam(models_v[m], X),
            exp_factor = exp.(out ./ S)

            exp_factor .* df_dx_subnet_reparam(models_v[m].layers[2][2], x2)
        end for m in 1:length(models_v)
    )

    dPsi_dJ = sum(
        let S = get_scale(raw_scale, m),
            out = eval_model_reparam(models_v[m], X),
            exp_factor = exp.(out ./ S)

            exp_factor .* df_dx_subnet_reparam(models_v[m].layers[2][3], x3)
        end for m in 1:length(models_v)
    )

    p0 = compute_p0_reparam(models_v, raw_scale)
    return vec(dPsi_dI1), vec(dPsi_dI2), vec(dPsi_dJ .- p0)
end

function eval_model_reparam(m_net, x)
    xs = split_input(x)
    sum(
        let net = m_net.layers[2][k]
            h1 = net.layers[1](xs[k])
            w2 = softplus.(net.layers[2].weight)
            b2 = net.layers[2].bias
            h2 = w2 * h1 .+ b2
            w3 = softplus(net.layers[3].weight[1])
            w3 .* h2
        end for k in 1:3
    )
end

function predict_all(models, raw_scale, x)
    p0 = compute_p0(models, raw_scale)
    psi_raw = sum(
        let S = get_scale(raw_scale, m)
            S .* (exp.(models[m](x) ./ S) .- exp.(models[m](X_REF) ./ S))
        end for m in 1:length(models)
    )
    J = x[3:3, :]
    return psi_raw .- p0 .* (J .- 1.0f0)
end

function predict_all_reparam(models, raw_scale, x)
    p0 = compute_p0_reparam(models, raw_scale)
    psi_raw = sum(
        let S = get_scale(raw_scale, m)
            out = eval_model_reparam(models[m], x)
            out_ref = eval_model_reparam(models[m], X_REF)
            S .* (exp.(out ./ S) .- exp.(out_ref ./ S))
        end for m in 1:length(models)
    )
    J = x[3:3, :]
    return psi_raw .- p0 .* (J .- 1.0f0)
end

# ----------------------------------------------------------------------
# 2b. Domain Ranges & Visualization Functions (from Neo_Hook_Model_Architecture_plot.jl)
# ----------------------------------------------------------------------
const X1_BOUNDS = (3.0, 12.0)
const X2_BOUNDS = (3.0, 10.0)
const X3_BOUNDS = (0.9, 1.8)

get_domain_ranges(num_points_per_dim) = (
    range(Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]), length=num_points_per_dim),
    range(Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]), length=num_points_per_dim),
    range(Float32(X3_BOUNDS[1]), Float32(X3_BOUNDS[2]), length=num_points_per_dim),
)

# Precompute 3D evaluation grid for mode share and norm computation
xs1_eval_grid, xs2_eval_grid, xs3_eval_grid = get_domain_ranges(15)
n1_eval, n2_eval, n3_eval = length(xs1_eval_grid), length(xs2_eval_grid), length(xs3_eval_grid)
x1_eval_grid_rep = repeat(xs1_eval_grid, inner=n2_eval * n3_eval)
x2_eval_grid_rep = repeat(repeat(xs2_eval_grid, inner=n3_eval), outer=n1_eval)
x3_eval_grid_rep = repeat(xs3_eval_grid, outer=n1_eval * n2_eval)
const X_3D_EVAL = [x1_eval_grid_rep'; x2_eval_grid_rep'; x3_eval_grid_rep']

"""
    plot_individual_nets_overlay(nets_x1, nets_x2, nets_x3, num_modes, raw_scale; n_pts=200, title_prefix="")

Overlay plot of individual sub-networks across all modes (as done in Neo_Hook_Model_Architecture_plot.jl).
"""
function plot_individual_nets_overlay(nets_x1, nets_x2, nets_x3, num_modes, raw_scale; n_pts=200, title_prefix="")
    xs1, xs2, xs3 = get_domain_ranges(n_pts)
    x1_mat = collect(xs1)'
    x2_mat = collect(xs2)'
    x3_mat = collect(xs3)'

    fig = Figure(size=(1650, 520), fontsize=20)

    t_prefix = isempty(title_prefix) ? "" : "($(title_prefix)) "
    ax1 = Axis(fig[1, 1], title="$(t_prefix)Individual Nets: I₁ across Modes", xlabel="I₁", ylabel="Subnet Contribution Δf",
        xgridvisible=true, ygridvisible=true)
    ax2 = Axis(fig[1, 2], title="$(t_prefix)Individual Nets: I₂ across Modes", xlabel="I₂", ylabel="Subnet Contribution Δf",
        xgridvisible=true, ygridvisible=true)
    ax3 = Axis(fig[1, 3], title="$(t_prefix)Individual Nets: J across Modes", xlabel="J", ylabel="Subnet Contribution Δf",
        xgridvisible=true, ygridvisible=true)

    colors = [:dodgerblue, :crimson, :darkorange, :purple, :forestgreen, :teal, :magenta]

    has_lbl1 = false
    has_lbl2 = false
    has_lbl3 = false

    for m in 1:num_modes
        w3_1 = nets_x1[m].layers[3].weight[1]
        w3_2 = nets_x2[m].layers[3].weight[1]
        w3_3 = nets_x3[m].layers[3].weight[1]

        ref1 = nets_x1[m](Float32[3.0;;])[1]
        ref2 = nets_x2[m](Float32[3.0;;])[1]
        ref3 = nets_x3[m](Float32[1.0;;])[1]

        df1 = vec(nets_x1[m](x1_mat)) .- ref1
        df2 = vec(nets_x2[m](x2_mat)) .- ref2
        df3 = vec(nets_x3[m](x3_mat)) .- ref3

        c = colors[mod1(m, length(colors))]

        if w3_1 > 0.001f0
            lines!(ax1, xs1, df1, color=c, linewidth=3.5, label="Mode $m (W₃=$(round(w3_1, digits=2)))")
            has_lbl1 = true
        end
        if w3_2 > 0.001f0
            lines!(ax2, xs2, df2, color=c, linewidth=3.5, label="Mode $m (W₃=$(round(w3_2, digits=2)))")
            has_lbl2 = true
        end
        if w3_3 > 0.001f0
            lines!(ax3, xs3, df3, color=c, linewidth=3.5, label="Mode $m (W₃=$(round(w3_3, digits=2)))")
            has_lbl3 = true
        end
    end

    has_lbl1 && axislegend(ax1, position=:lt, framevisible=true)
    has_lbl2 && axislegend(ax2, position=:lt, framevisible=true)
    has_lbl3 && axislegend(ax3, position=:lt, framevisible=true)

    return fig
end

"""
    plot_1d_subnetworks(models, raw_scale, nets_x1, nets_x2, nets_x3, num_modes; n_pts=200, iter=nothing, opt_label="")

Table grid plot of 1D sub-network outputs per mode (matching D3 `f_nets` in visualize_results).
Each column represents a mode, with rows for Net x1 (I₁), Net x2 (I₂), Net x3 (J),
and mode share & scale summary at the bottom.
"""
function plot_1d_subnetworks(models, raw_scale, nets_x1, nets_x2, nets_x3, num_modes; n_pts=200, iter=nothing, opt_label="")
    iter_label = iter !== nothing ? (isempty(opt_label) ? " (Iteration $iter)" : " (Iteration $iter $opt_label)") : ""
    xs1_net, xs2_net, xs3_net = get_domain_ranges(n_pts)
    x1_net = collect(xs1_net)'
    x2_net = collect(xs2_net)'
    x3_net = collect(xs3_net)'

    f_nets = Figure(size=(max(1000, 280 * num_modes), 970))
    Label(f_nets[0, :], "Sub-network outputs per mode$iter_label", fontsize=16, font=:bold)

    mode_preds_global = [
        let S = get_scale(raw_scale, k)
            vec(S .* (exp.(models[k](X_3D_EVAL) ./ S) .- exp.(models[k](X_REF) ./ S)))
        end for k in 1:num_modes
    ]
    mode_norms_global = [norm(mp, 2) for mp in mode_preds_global]
    sum_norms_global = sum(mode_norms_global)

    ref_x1 = Float32[3.0f0;;]
    ref_x2 = Float32[3.0f0;;]
    ref_x3 = Float32[1.0f0;;]

    p0_val = Float32(compute_p0(models, raw_scale))

    for mode in 1:num_modes
        S_mode = get_scale(raw_scale, mode)
        net1_out = vec(S_mode .* (exp.(nets_x1[mode](x1_net) ./ S_mode) .- exp.(nets_x1[mode](ref_x1) ./ S_mode)))
        net2_out = vec(S_mode .* (exp.(nets_x2[mode](x2_net) ./ S_mode) .- exp.(nets_x2[mode](ref_x2) ./ S_mode)))
        net3_out = vec(S_mode .* (exp.(nets_x3[mode](x3_net) ./ S_mode) .- exp.(nets_x3[mode](ref_x3) ./ S_mode)))

        w3_1 = nets_x1[mode].layers[3].weight[1]
        w3_2 = nets_x2[mode].layers[3].weight[1]
        w3_3 = nets_x3[mode].layers[3].weight[1]

        ax1 = Axis(f_nets[1, mode],
            title="Mode $mode – Net I₁",
            xlabel="I₁",
            ylabel="output")
        lines!(ax1, xs1_net, net1_out, color=:blue, linewidth=3.5, label="Ψ(I₁)")
        if w3_1 > 0.001f0
            ref_c10 = (Float32(μ_true) / 2.0f0) .* (xs1_net .- 3.0f0)
            lines!(ax1, xs1_net, ref_c10, color=:crimson, linestyle=:dot, linewidth=4, label="μ/2(I₁-3)")
        else
            ylims!(ax1, -1.0, 1.0)
        end
        axislegend(ax1, position=:lt)

        ax2 = Axis(f_nets[2, mode],
            title="Mode $mode – Net I₂",
            xlabel="I₂",
            ylabel="output")
        lines!(ax2, xs2_net, net2_out, color=:orange, linewidth=3.5, label="Ψ(I₂)")
        if w3_2 <= 0.001f0
            ylims!(ax2, -1.0, 1.0)
        end
        axislegend(ax2, position=:lt)

        ax3 = Axis(f_nets[3, mode],
            title="Mode $mode – Net J",
            xlabel="J",
            ylabel="output")
        if w3_3 > 0.001f0
            scaled3_phys = net3_out .- p0_val .* (xs3_net .- 1.0f0)
            lines!(ax3, xs3_net, scaled3_phys, color=:forestgreen, linewidth=3.5, label="Ψ(J)")
            ref_vol_true = @. -Float32(μ_true) * log(max(1f-6, xs3_net)) + (Float32(λ_true) / 2.0f0) * (xs3_net - 1.0f0)^2
            lines!(ax3, xs3_net, ref_vol_true, color=:crimson, linestyle=:dot, linewidth=4, label="-μ ln(J) + λ/2(J-1)²")
            ref_vol_pure = (Float32(λ_true) / 2.0f0) .* (xs3_net .- 1.0f0) .^ 2
            lines!(ax3, xs3_net, ref_vol_pure, color=:gray60, linestyle=:dash, linewidth=2, label="λ/2(J-1)²")
            axislegend(ax3, position=:rt)
        else
            lines!(ax3, xs3_net, net3_out, color=:forestgreen, linewidth=3.5, label="Ψ(J)")
            ylims!(ax3, -1.0, 1.0)
            axislegend(ax3, position=:lt)
        end

        mode_share_global = mode_norms_global[mode] / max(sum_norms_global, 1f-8)
        Label(f_nets[4, mode], "scale = $(round(S_mode, digits=2))\nMode Share = $(round(mode_share_global * 100, digits=2))%", fontsize=13, font=:bold)
    end

    rowsize!(f_nets.layout, 4, Fixed(45))

    for col in 1:num_modes
        colsize!(f_nets.layout, col, Relative(1 / num_modes))
    end

    return f_nets
end

# ----------------------------------------------------------------------
# 3. EUCLID Unsupervised Loss Function with Sparsity Penalties
# ----------------------------------------------------------------------
println("\n[3/6] Setting up EUCLID Unsupervised Loss with L1 & Intra-Mode Regularization...")

const alpha_react = 50.0f0
const last_euclid_data_loss = Ref{Float32}(0.0f0)

function loss_euclid_simultaneous(models, raw_scale, lam, lam_cross)
    total_loss = 0.0f0
    for t in 1:n_steps
        d1, d2, dJ = eval_derivatives_all(models, raw_scale, I1_f32[t], I2_f32[t], J_f32[t])
        g_int = K1_f32[t] * d1 .+ K2_f32[t] * d2 .+ K3_f32[t] * dJ

        F_target = F_ext_f32[t]
        r_free = g_int[free_dofs]
        l_equil = sum(abs2, r_free) / (F_target^2 + 1.0f0)

        F_pred = sum(g_int[pull_dofs_x])
        l_react = abs2((F_pred - F_target) / F_target)

        total_loss += (l_equil + alpha_react * l_react)
    end
    euclid_term = total_loss / Float32(n_steps)
    Flux.Zygote.ignore() do
        last_euclid_data_loss[] = Float32(euclid_term)
    end

    # Regularization penalties matching D3
    reg_l1 = zero(Float32)
    reg_intra = zero(Float32)
    for m in models
        w1 = m.layers[2][1][3].weight[1]
        w2 = m.layers[2][2][3].weight[1]
        w3 = m.layers[2][3][3].weight[1]

        reg_l1 += w1 + w2 + w3
        reg_intra += w1 * w2 * w3
    end

    return euclid_term + lam * reg_l1 + lam_cross * reg_intra
end

function loss_euclid_reparam(models_v, raw_scale)
    total_loss = 0.0f0
    for t in 1:n_steps
        d1, d2, dJ = eval_derivatives_all_reparam(models_v, raw_scale, I1_f32[t], I2_f32[t], J_f32[t])
        g_int = K1_f32[t] * d1 .+ K2_f32[t] * d2 .+ K3_f32[t] * dJ

        F_target = F_ext_f32[t]
        r_free = g_int[free_dofs]
        l_equil = sum(abs2, r_free) / (F_target^2 + 1.0f0)

        F_pred = sum(g_int[pull_dofs_x])
        l_react = abs2((F_pred - F_target) / F_target)

        total_loss += (l_equil + alpha_react * l_react)
    end
    return total_loss / Float32(n_steps)
end

# ----------------------------------------------------------------------
# 4. Training Hyperparameters & Initialization
# ----------------------------------------------------------------------
num_modes = 5
vol_mode_idx = num_modes >= 5 ? 5 : 0
eta = 0.05f0               # Adam learning rate
lam = 0.01f0              # L1 regularization strength
lam_cross_init = 0.0f0     # initial cross-term penalty
lam_cross_max = 1.0f0      # target cross-term penalty
n_epochs = 50              # inner epochs per checkpoint
num_iterations = 60        # total checkpoints (adaptive for speed & convergence)
iter_max_lam = 10          # ramp-up checkpoint for penalties
dedup_start_iteration = 10 # iteration at which mode deduplication and combination begins
lbfgs_switch_iteration = 20 # checkpoint to switch to L-BFGS
lbfgs_iters = 50           # L-BFGS iterations per checkpoint
min_remaining_modes = 1    # minimum modes to retain
share_thresh = 0.005f0     # threshold for pruning inactive modes
initial_num_modes = num_modes
noise_level = 0.02f0  # Relative noise standard deviation (e.g., 0.02 = 2%, 0.05 = 5%)

# Generate static noisy strain invariants once for the entire training run
noisy_I1, noisy_I2, noisy_J = add_dic_strain_noise(I1_clean, I2_clean, J_clean, noise_level)
const I1_f32 = noisy_I1
const I2_f32 = noisy_I2
const J_f32 = noisy_J

nets_x1, nets_x2, nets_x3, models = init_model(num_modes)

init_S = 20.0f0
init_raw = inv_sp(init_S)
raw_scale = Float32[init_raw]
residual_over_iteration = Float32[]       # pure EUCLID data loss (without regularization)
total_loss_over_iteration = Float32[]     # total loss (with L1 & intra-mode regularization)

params = (models=models, scale=raw_scale)
opt = Flux.setup(Flux.Adam(eta), params)

println("Initial EUCLID loss: ", loss_euclid_simultaneous(models, raw_scale, 0.0f0, 0.0f0))

# ----------------------------------------------------------------------
# 5. Hybrid Training Loop (Adam + Proximal L1 + L-BFGS)
# ----------------------------------------------------------------------
script_name = "EUCLID_SparseDecomp"
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
noise_pct_str = "$(round(Int, noise_level * 100))pct"
save_dir = datadir("exp_raw", "$(script_name)_$(noise_pct_str)_$(timestamp)")
iter_plots_dir = joinpath(save_dir, "iterations")
mkpath(iter_plots_dir)
mkpath(plotsdir("Visualizations"))

println("\n[4/6] Starting Hybrid Unsupervised Training: $num_iterations checkpoints...")
println("      (Adam for 1..$lbfgs_switch_iteration, L-BFGS for $(lbfgs_switch_iteration + 1)..$num_iterations)\n")
println("      Output directory: $save_dir")
println("      Plotting sub-network table every 2 iterations to: $iter_plots_dir")
println("      Saving network checkpoints every 10 iterations to: $save_dir\n")

# Initial sub-network table before training (Iteration 0, matching D3)
fig_init = plot_1d_subnetworks(models, raw_scale, nets_x1, nets_x2, nets_x3, num_modes; iter=0)
display(fig_init)
save(joinpath(iter_plots_dir, "subnetworks_table_iter_00.png"), fig_init)
save(plotsdir("Visualizations", "latest_subnetworks_table.png"), fig_init)

ramp_epochs = max(1, (iter_max_lam - 1) * n_epochs)

for iter_num in 1:num_iterations
    local current_lam, current_lam_cross

    if iter_num <= lbfgs_switch_iteration
        # --- Stage 1: Adam with Proximal Regularization ---
        for sub_epoch in 1:n_epochs
            epoch = (iter_num - 1) * n_epochs + sub_epoch
            progress = clamp(Float32(epoch - 1) / Float32(ramp_epochs), 0.0f0, 1.0f0)
            current_lam_cross = lam_cross_init + (lam_cross_max - lam_cross_init) * progress
            current_lam = lam * progress

            val, grads = Flux.withgradient(p -> loss_euclid_simultaneous(p.models, p.scale, current_lam, current_lam_cross), params)
            Flux.update!(opt, params, grads[1])

            # ICNN projection: enforce W2 >= 1e-4 and proximal operator for W3:
            for m in models
                m.layers[2][1][2].weight .= max.(1.0f-4, m.layers[2][1][2].weight)
                m.layers[2][2][2].weight .= max.(1.0f-4, m.layers[2][2][2].weight)
                m.layers[2][3][2].weight .= max.(1.0f-4, m.layers[2][3][2].weight)

                m.layers[2][1][3].weight[1] = max(0.0f0, m.layers[2][1][3].weight[1] - current_lam * eta)
                m.layers[2][2][3].weight[1] = max(0.0f0, m.layers[2][2][3].weight[1] - current_lam * eta)
                m.layers[2][3][3].weight[1] = max(0.0f0, m.layers[2][3][3].weight[1] - current_lam * eta)
            end

            # Keep volumetric mode strictly dedicated to J: freeze x1 and x2 at 0
            if 1 <= vol_mode_idx <= length(models)
                models[vol_mode_idx].layers[2][1][3].weight .= 0.0f0
                models[vol_mode_idx].layers[2][2][3].weight .= 0.0f0
            end

            push!(residual_over_iteration, last_euclid_data_loss[])
            push!(total_loss_over_iteration, Float32(val))
        end
    else
        # --- Stage 2: Reparameterized L-BFGS Fine-Tuning ---
        if iter_num == lbfgs_switch_iteration + 1
            printstyled("\n" * "="^75 * "\n", color=:green, bold=true)
            printstyled(">>> Switching to Reparameterized L-BFGS optimization at checkpoint $iter_num\n", color=:green, bold=true)
            printstyled(">>> Enforcing W2, W3 >= 0 via softplus reparameterization.\n", color=:green, bold=true)
            printstyled("="^75 * "\n", color=:green, bold=true)

            global models_v = deepcopy(models)
            for m in 1:num_modes
                for k in 1:3
                    subnet = models_v[m].layers[2][k]
                    subnet.layers[2].weight .= inv_sp(subnet.layers[2].weight)
                    w3_val = models[m].layers[2][k].layers[3].weight[1]
                    if w3_val > 0.001f0 && !(m == vol_mode_idx && k != 3)
                        subnet.layers[3].weight .= inv_sp(subnet.layers[3].weight)
                    else
                        subnet.layers[3].weight .= -20.0f0
                    end
                end
            end

            frozen_mask_models = deepcopy(models_v)
            for m in 1:num_modes
                for k in 1:3
                    subnet = frozen_mask_models[m].layers[2][k]
                    subnet.layers[1].weight .= 1.0f0
                    subnet.layers[1].bias .= 1.0f0
                    subnet.layers[2].weight .= 1.0f0
                    subnet.layers[2].bias .= 1.0f0
                    w3_val = models[m].layers[2][k].layers[3].weight[1]
                    if w3_val > 0.001f0 && !(m == vol_mode_idx && k != 3)
                        subnet.layers[3].weight .= 1.0f0
                    else
                        subnet.layers[3].weight .= 0.0f0
                    end
                end
            end
            global frozen_mask_flat, _ = Flux.destructure((models=frozen_mask_models, scale=Float32[1.0f0]))
        end

        current_lam = 0.0f0
        current_lam_cross = 0.0f0

        theta_v, reconstruct_v = Flux.destructure((models=models_v, scale=raw_scale))

        let last_th = nothing, last_f = 0.0f0, last_g = similar(theta_v)
            function eval_fg!(th)
                if last_th != th
                    last_th = copy(th)
                    curr_params = reconstruct_v(th)
                    val, grads = Flux.withgradient(p -> loss_euclid_reparam(p.models, p.scale), curr_params)
                    g_flat, _ = Flux.destructure(grads[1])
                    for i in 1:length(g_flat)
                        if frozen_mask_flat[i] == 0.0f0
                            g_flat[i] = 0.0f0
                        end
                    end
                    last_f = Float32(val)
                    last_g .= g_flat
                end
            end
            f_obj(th) = (eval_fg!(th); last_f)
            g_obj!(G, th) = (eval_fg!(th); G .= last_g)

            opt_res = Optim.optimize(
                f_obj, g_obj!, theta_v,
                Optim.LBFGS(),
                Optim.Options(iterations=lbfgs_iters)
            )

            final_v = reconstruct_v(Optim.minimizer(opt_res))
            global models_v = final_v.models
            global raw_scale = final_v.scale

            for m in 1:num_modes
                for k in 1:3
                    sub_v = models_v[m].layers[2][k]
                    sub_m = models[m].layers[2][k]
                    sub_m.layers[1].weight .= sub_v.layers[1].weight
                    sub_m.layers[1].bias .= sub_v.layers[1].bias
                    sub_m.layers[2].weight .= softplus.(sub_v.layers[2].weight)
                    sub_m.layers[2].bias .= sub_v.layers[2].bias
                    if sub_v.layers[3].weight[1] > -15.0f0 && !(m == vol_mode_idx && k != 3)
                        sub_m.layers[3].weight .= softplus.(sub_v.layers[3].weight)
                    else
                        sub_m.layers[3].weight .= 0.0f0
                    end
                end
            end
            global params = (models=models, scale=raw_scale)

            # Keep nets_x1, nets_x2, nets_x3 synced
            for i in 1:num_modes
                nets_x1[i] = models[i].layers[2][1]
                nets_x2[i] = models[i].layers[2][2]
                nets_x3[i] = models[i].layers[2][3]
            end

            lbfgs_loss = Float32(Optim.minimum(opt_res))
            push!(residual_over_iteration, lbfgs_loss)
            push!(total_loss_over_iteration, lbfgs_loss)
        end
    end

    opt_label = iter_num > lbfgs_switch_iteration ? "[L-BFGS]" : "[Adam]"
    curr_loss = residual_over_iteration[end]
    p0_curr = compute_p0(models, raw_scale)
    println("Checkpoint $(rpad(iter_num, 2)) / $num_iterations $opt_label: Loss = $(round(curr_loss, digits=5)), Scale S = $(round(get_scale(raw_scale, 1), digits=2)), p₀ = $(round(p0_curr, digits=4)) (stress-free)")

    # Mode share analysis across evaluation domain
    mode_preds = [
        let S = get_scale(raw_scale, j)
            vec(S .* (exp.(models[j](X_3D_EVAL) ./ S) .- exp.(models[j](X_REF) ./ S)))
        end for j in 1:num_modes
    ]
    mode_norms = [norm(mp, 2) for mp in mode_preds]
    sum_norms = sum(mode_norms)
    for j in 1:num_modes
        mode_share = mode_norms[j] / max(sum_norms, 1f-8)
        println("   • Mode $j Relative Share: $(round(mode_share * 100, digits=2))%")
    end

    # Mode combination and deduplication based on matching active sub-networks (exact D3 strategy)
    mode_removed_this_iter = false
    if iter_num <= lbfgs_switch_iteration && iter_num >= dedup_start_iteration && num_modes > min_remaining_modes
        net_lists = [nets_x1, nets_x2, nets_x3]
        net_names = ["I₁", "I₂", "J"]

        xs1_eval, xs2_eval, xs3_eval = get_domain_ranges(200)
        x_evals = [collect(xs1_eval)', collect(xs2_eval)', collect(xs3_eval)']

        sim_found = false
        target_remove = 0
        survivor = 0
        best_active_net_str = ""

        ref_evals = [Float32[3.0f0;;], Float32[3.0f0;;], Float32[1.0f0;;]]
        for m1 in 1:num_modes
            S_m1 = get_scale(raw_scale, m1)
            sub_outs1 = [vec(S_m1 .* (exp.(net_lists[k][m1](x_evals[k]) ./ S_m1) .- exp.(net_lists[k][m1](ref_evals[k]) ./ S_m1))) for k in 1:3]
            stds1 = [std(so) for so in sub_outs1]
            w_thresh = 0.005f0
            std_thresh = 0.01f0
            active1 = [net_lists[k][m1].layers[3].weight[1] > w_thresh && stds1[k] > std_thresh for k in 1:3]
            if !any(active1)
                continue
            end

            for m2 in (m1+1):num_modes
                S_m2 = get_scale(raw_scale, m2)
                sub_outs2 = [vec(S_m2 .* (exp.(net_lists[l][m2](x_evals[l]) ./ S_m2) .- exp.(net_lists[l][m2](ref_evals[l]) ./ S_m2))) for l in 1:3]
                stds2 = [std(so) for so in sub_outs2]
                active2 = [net_lists[l][m2].layers[3].weight[1] > w_thresh && stds2[l] > std_thresh for l in 1:3]

                # Directly merge if EXACTLY the same sub-networks are active
                if active1 == active2 && any(active1)
                    if m1 == vol_mode_idx
                        survivor = m1
                        target_remove = m2
                    elseif m2 == vol_mode_idx
                        survivor = m2
                        target_remove = m1
                    elseif mode_norms[m1] >= mode_norms[m2]
                        survivor = m1
                        target_remove = m2
                    else
                        survivor = m2
                        target_remove = m1
                    end

                    # Add linear scaling weights on active channels into survivor
                    for k in 1:3
                        if active1[k]
                            models[survivor].layers[2][k][3].weight[1] += models[target_remove].layers[2][k][3].weight[1]
                        end
                    end

                    best_active_net_str = join(net_names[active1], ", ")
                    share_survivor = round(mode_norms[survivor] / max(sum_norms, 1f-8) * 100, digits=2)
                    share_remove = round(mode_norms[target_remove] / max(sum_norms, 1f-8) * 100, digits=2)
                    printstyled("Modes $m1 and $m2 have the exact same active nets [$best_active_net_str]! Adding duplicate Mode $target_remove (share: $share_remove%) to Mode $survivor (share: $share_survivor%).\n", color=:yellow, bold=true)
                    sim_found = true
                    break
                end
            end
            if sim_found
                break
            end
        end

        # Priority 2: Subspace / Subset match (1D mode in I1 or I2 added to 2D mode containing that channel)
        # e.g. Mode with I1 only added to Mode with (I1, J)
        if !sim_found
            for m1 in 1:num_modes
                if m1 == vol_mode_idx
                    continue
                end
                S_m1 = get_scale(raw_scale, m1)
                sub_outs1 = [vec(S_m1 .* (exp.(net_lists[k][m1](x_evals[k]) ./ S_m1) .- exp.(net_lists[k][m1](ref_evals[k]) ./ S_m1))) for k in 1:3]
                stds1 = [std(so) for so in sub_outs1]
                w_thresh = 0.005f0
                std_thresh = 0.01f0
                active1 = [net_lists[k][m1].layers[3].weight[1] > w_thresh && stds1[k] > std_thresh for k in 1:3]

                if count(active1) == 1
                    shared_k = findfirst(active1)
                    if shared_k in (1, 2) # only isochoric coordinates I1 or I2
                        for m2 in 1:num_modes
                            if m1 == m2 || m2 == vol_mode_idx
                                continue
                            end
                            S_m2 = get_scale(raw_scale, m2)
                            sub_outs2 = [vec(S_m2 .* (exp.(net_lists[l][m2](x_evals[l]) ./ S_m2) .- exp.(net_lists[l][m2](ref_evals[l]) ./ S_m2))) for l in 1:3]
                            stds2 = [std(so) for so in sub_outs2]
                            active2 = [net_lists[l][m2].layers[3].weight[1] > w_thresh && stds2[l] > std_thresh for l in 1:3]

                            # m1 is 1D active in I1 or I2, and m2 is a 2D mode containing that same channel
                            if count(active2) >= 2 && active2[shared_k]
                                survivor = m2
                                target_remove = m1

                                # Add linear scaling weight of shared channel into survivor
                                models[survivor].layers[2][shared_k][3].weight[1] += models[target_remove].layers[2][shared_k][3].weight[1]

                                share_survivor = round(mode_norms[survivor] / max(sum_norms, 1f-8) * 100, digits=2)
                                share_remove = round(mode_norms[target_remove] / max(sum_norms, 1f-8) * 100, digits=2)
                                sup_str = join(net_names[active2], ", ")
                                printstyled("Mode $target_remove is a 1D subset [$(net_names[shared_k])] of Mode $survivor [$sup_str]! Adding Mode $target_remove (share: $share_remove%) to Mode $survivor (share: $share_survivor%).\n", color=:yellow, bold=true)
                                sim_found = true
                                break
                            end
                        end
                    end
                end
                if sim_found
                    break
                end
            end
        end

        if sim_found && num_modes > min_remaining_modes
            # Keep the exact values of the mode with higher mode share (survivor) and remove the other mode
            deleteat!(models, target_remove)
            deleteat!(nets_x1, target_remove)
            deleteat!(nets_x2, target_remove)
            deleteat!(nets_x3, target_remove)
            if length(raw_scale) > 1
                deleteat!(raw_scale, target_remove)
            end
            if target_remove == vol_mode_idx
                global vol_mode_idx = 0
            elseif target_remove < vol_mode_idx
                global vol_mode_idx -= 1
            end
            if 1 <= vol_mode_idx <= length(models)
                models[vol_mode_idx].layers[2][1][3].weight .= 0.0f0
                models[vol_mode_idx].layers[2][2][3].weight .= 0.0f0
            end
            global num_modes -= 1
            for i in 1:num_modes
                nets_x1[i] = models[i].layers[2][1]
                nets_x2[i] = models[i].layers[2][2]
                nets_x3[i] = models[i].layers[2][3]
            end
            global params = (models=models, scale=raw_scale)
            global opt = Flux.setup(Flux.Adam(eta), params)
            mode_removed_this_iter = true
            printstyled("Total number of modes lowered to $num_modes.\n", color=:yellow, bold=true)
        elseif !sim_found && num_modes > min_remaining_modes
            # Inactive / low energy share mode pruning (removes idle modes with < share_thresh share or no active subnets)
            for j in 1:num_modes
                if j == vol_mode_idx
                    continue # Do not prune dedicated volumetric mode
                end
                mode_share = mode_norms[j] / max(sum_norms, 1f-8)
                sub_outs_j = [vec(get_scale(raw_scale, j) .* (exp.(net_lists[k][j](x_evals[k]) ./ get_scale(raw_scale, j)) .- exp.(net_lists[k][j](ref_evals[k]) ./ get_scale(raw_scale, j)))) for k in 1:3]
                stds_j = [std(so) for so in sub_outs_j]
                w3_j = [net_lists[k][j].layers[3].weight[1] for k in 1:3]
                act_j = [w3_j[k] > 0.005f0 && stds_j[k] > 0.01f0 for k in 1:3]

                if (!any(act_j) || mode_share < share_thresh) && num_modes > min_remaining_modes
                    reason = !any(act_j) ? "inactive subnets" : "share $(round(mode_share * 100, digits=2))% < $(round(share_thresh * 100, digits=1))%"
                    printstyled("Mode $j has $reason. Removing idle mode!\n", color=:magenta, bold=true)
                    deleteat!(models, j)
                    deleteat!(nets_x1, j)
                    deleteat!(nets_x2, j)
                    deleteat!(nets_x3, j)
                    if length(raw_scale) > 1
                        deleteat!(raw_scale, j)
                    end
                    if j == vol_mode_idx
                        global vol_mode_idx = 0
                    elseif j < vol_mode_idx
                        global vol_mode_idx -= 1
                    end
                    if 1 <= vol_mode_idx <= length(models)
                        models[vol_mode_idx].layers[2][1][3].weight .= 0.0f0
                        models[vol_mode_idx].layers[2][2][3].weight .= 0.0f0
                    end
                    global num_modes -= 1
                    for i in 1:num_modes
                        nets_x1[i] = models[i].layers[2][1]
                        nets_x2[i] = models[i].layers[2][2]
                        nets_x3[i] = models[i].layers[2][3]
                    end
                    global params = (models=models, scale=raw_scale)
                    global opt = Flux.setup(Flux.Adam(eta), params)
                    mode_removed_this_iter = true
                    printstyled("Total number of modes lowered to $num_modes.\n", color=:magenta, bold=true)
                    break
                end
            end
        end

        if mode_removed_this_iter
            loss_val = Float32(loss_euclid_simultaneous(models, raw_scale, current_lam, current_lam_cross))
            residual_over_iteration[end] = last_euclid_data_loss[]
            total_loss_over_iteration[end] = loss_val
        end
    end

    # Plot sub-network table after every 2 iterations (and at the final iteration, matching D3)
    if iter_num % 2 == 0 || iter_num == num_iterations
        nets_x1_curr = [models[i].layers[2][1] for i in 1:num_modes]
        nets_x2_curr = [models[i].layers[2][2] for i in 1:num_modes]
        nets_x3_curr = [models[i].layers[2][3] for i in 1:num_modes]

        fig_table = plot_1d_subnetworks(models, raw_scale, nets_x1_curr, nets_x2_curr, nets_x3_curr, num_modes;
            iter=iter_num, opt_label=opt_label)
        display(fig_table)
        iter_plot_file = joinpath(iter_plots_dir, "subnetworks_table_iter_$(lpad(iter_num, 2, "0")).png")
        save(iter_plot_file, fig_table)
        save(plotsdir("Visualizations", "latest_subnetworks_table.png"), fig_table)
        println("   [Visualization] Saved sub-network table plot to: $iter_plot_file")
    end

    # Save network checkpoint every 10 iterations with a new name in the save folder
    if iter_num % 10 == 0
        iter_network_file = joinpath(save_dir, "network_iter_$(lpad(iter_num, 2, "0")).jld2")
        learned_F_curr = Float32[]
        for t in 1:n_steps
            d1, d2, dJ = eval_derivatives_all(models, raw_scale, I1_f32[t], I2_f32[t], J_f32[t])
            g_int = K1_f32[t] * d1 .+ K2_f32[t] * d2 .+ K3_f32[t] * dJ
            push!(learned_F_curr, sum(g_int[pull_dofs_x]))
        end

        checkpoint_data = Dict(
            "models" => deepcopy(models),
            "raw_scale" => copy(raw_scale),
            "initial_num_modes" => initial_num_modes,
            "num_modes" => num_modes,
            "vol_mode_idx" => vol_mode_idx,
            "p0" => p0_curr,
            "step_indices" => step_indices,
            "learned_F" => learned_F_curr,
            "F_true" => F_ext_f32,
            "F_ext_history" => F_ext_history,
            "loss_history" => copy(residual_over_iteration),
            "total_loss_history" => copy(total_loss_over_iteration),
            "dedup_start_iteration" => dedup_start_iteration,
            "share_thresh" => share_thresh,
            "iteration" => iter_num,
            "noise_level" => noise_level
        )
        wsave(iter_network_file, checkpoint_data)
        println("   [Checkpoint] Saved network checkpoint to: $iter_network_file")
    end
end

println("\n" * "="^75)
println("Training completed successfully!")
println("="^75)

# ----------------------------------------------------------------------
# 6. Evaluation, Model Verification & Plotting
# ----------------------------------------------------------------------
println("\n[5/6] Evaluating Identified Constitutive Model...")

# 1. Reaction Force Comparison across Selected Load Steps
println("\nReaction Force Comparison ($n_steps Selected Steps):")
println("Step |  FE Experimental (N) |  Identified PGD-NN (N) |  Rel Error (%)")
println("-"^70)

learned_F = Float32[]
for t in 1:n_steps
    d1, d2, dJ = eval_derivatives_all(models, raw_scale, I1_f32[t], I2_f32[t], J_f32[t])
    g_int = K1_f32[t] * d1 .+ K2_f32[t] * d2 .+ K3_f32[t] * dJ
    F_pred = sum(g_int[pull_dofs_x])
    push!(learned_F, F_pred)
    err = abs(F_pred - F_ext_f32[t]) / F_ext_f32[t] * 100.0f0
    println(rpad(step_indices[t], 5), "|  ", rpad(round(F_ext_f32[t], digits=3), 21), "|  ", rpad(round(F_pred, digits=3), 23), "|  ", round(err, digits=2), "%")
end

# Compute prediction across all total steps to evaluate interpolation / extrapolation
println("\nGeneralization Evaluation across all $n_steps_total FE Steps:")
learned_F_all = Float32[]
for t in 1:n_steps_total
    K1_t = SparseMatrixCSC{Float32,Int}(K1_history[t])
    K2_t = SparseMatrixCSC{Float32,Int}(K2_history[t])
    K3_t = SparseMatrixCSC{Float32,Int}(K3_history[t])
    d1, d2, dJ = eval_derivatives_all(models, raw_scale, Float32.(I1_history[t]), Float32.(I2_history[t]), Float32.(J_history[t]))
    g_int = K1_t * d1 .+ K2_t * d2 .+ K3_t * dJ
    push!(learned_F_all, sum(g_int[pull_dofs_x]))
end
mean_rel_err_all = mean(abs.(learned_F_all .- Float32.(F_ext_history)) ./ Float32.(F_ext_history)) * 100.0f0
println("  • Mean Relative Reaction Force Error (All $n_steps_total Steps): $(round(mean_rel_err_all, digits=2))%")

# 2. Strain Energy Comparison against Ground Truth
I1_eval = Float32.(range(3.01, 5.0, length=50))
I2_eval = Float32.(range(3.01, 5.0, length=50))
J_eval = Float32.(range(0.9, 1.2, length=50))
eval_mat = [I1_eval'; I2_eval'; J_eval']

psi_pred = vec(predict_all(models, raw_scale, eval_mat))
psi_true = [Ψ_true(I1_eval[i], I2_eval[i], J_eval[i]) for i in 1:50]

r2_score = 1.0 - sum(abs2, psi_pred .- psi_true) / sum(abs2, psi_true .- mean(psi_true))
println("\nStrain Energy Identification Metric:")
println("  • R² Score against True Neo-Hookean Energy: ", round(r2_score, digits=4))

# 3. Stress-Free Reference Configuration Verification
p0_final = compute_p0(models, raw_scale)
d1_0, d2_0, dJ_0 = eval_derivatives_all(models, raw_scale, Float32[3.0], Float32[3.0], Float32[1.0])
P_ref = 2.0f0 * d1_0[1] + 4.0f0 * d2_0[1] + dJ_0[1]
println("\nStress-Free Reference Configuration Verification:")
println("  • Raw Reference Hydrostatic Pressure p₀ = ", round(p0_final, digits=5))
println("  • Corrected Reference Stress P(I) = ", round(P_ref, digits=6), " (identically zero: $(abs(P_ref) < 1e-5))")

# 4. Convexity Verification
println("\nConvexity Verification (Layer 2 & Layer 3 Weights):")
for m in 1:num_modes
    w2_1 = minimum(models[m].layers[2][1].layers[2].weight)
    w3_1 = models[m].layers[2][1].layers[3].weight[1]
    w2_2 = minimum(models[m].layers[2][2].layers[2].weight)
    w3_2 = models[m].layers[2][2].layers[3].weight[1]
    w2_3 = minimum(models[m].layers[2][3].layers[2].weight)
    w3_3 = models[m].layers[2][3].layers[3].weight[1]
    println("Mode $m: W2_min = $(round(min(w2_1, w2_2, w2_3), digits=4)), W3 = [$(round(w3_1, digits=4)), $(round(w3_2, digits=4)), $(round(w3_3, digits=4))]")
end

# 5. Save Network and Results
println("\n[6/6] Saving Identified Model and Generating Plots...")

save_data = Dict(
    "models" => models,
    "raw_scale" => raw_scale,
    "initial_num_modes" => initial_num_modes,
    "num_modes" => num_modes,
    "vol_mode_idx" => vol_mode_idx,
    "p0" => p0_final,
    "step_indices" => step_indices,
    "learned_F" => learned_F,
    "learned_F_all" => learned_F_all,
    "F_true" => F_ext_f32,
    "F_ext_history" => F_ext_history,
    "r2_score" => r2_score,
    "loss_history" => residual_over_iteration,
    "total_loss_history" => total_loss_over_iteration,
    "dedup_start_iteration" => dedup_start_iteration,
    "share_thresh" => share_thresh,
    "iteration" => num_iterations,
    "noise_level" => noise_level,
    "formulation" => "d3"
)
save_path = joinpath(save_dir, "network.jld2")
wsave(save_path, save_data)
println("Saved model checkpoint to: ", save_path)

# Save hyperparameters and experiment configuration to parameters.txt (matching D3)
params_txt_path = joinpath(save_dir, "parameters.txt")
open(params_txt_path, "w") do io
    println(io, "============================================================")
    println(io, "EUCLID Hyperparameters & Experiment Configuration")
    println(io, "Timestamp: $timestamp")
    println(io, "Script: $script_name")
    println(io, "============================================================")
    println(io, "initial_num_modes          = $initial_num_modes")
    println(io, "final_num_modes            = $num_modes")
    println(io, "vol_mode_idx               = $vol_mode_idx")
    println(io, "eta                        = $eta")
    println(io, "lam                        = $lam")
    println(io, "lam_cross_init             = $lam_cross_init")
    println(io, "lam_cross_max              = $lam_cross_max")
    println(io, "n_epochs                   = $n_epochs")
    println(io, "num_iterations             = $num_iterations")
    println(io, "iter_max_lam               = $iter_max_lam")
    println(io, "dedup_start_iteration      = $dedup_start_iteration")
    println(io, "lbfgs_switch_iteration     = $lbfgs_switch_iteration")
    println(io, "lbfgs_iters                = $lbfgs_iters")
    println(io, "min_remaining_modes        = $min_remaining_modes")
    println(io, "share_thresh               = $share_thresh")
    println(io, "p0_ref                     = $p0_final")
    println(io, "learned_scale_S            = $(round(get_scale(raw_scale, 1), digits=4))")
    println(io, "r2_score                   = $r2_score")
    println(io, "noise_level                = $noise_level")
    println(io, "============================================================")
end
println("Saved parameters text to: ", params_txt_path)

# Generate Diagnostic Figures with CairoMakie
fig = Figure(size=(1100, 480))

# Subplot 1: Loss Evolution
ax1 = Axis(fig[1, 1], xlabel="Epochs", ylabel="Loss", title="Unsupervised Training Convergence", yscale=log10)
lines!(ax1, 1:length(residual_over_iteration), residual_over_iteration, color=:dodgerblue, linewidth=2, label="Pure EUCLID Data Loss")
lines!(ax1, 1:length(total_loss_over_iteration), total_loss_over_iteration, color=:gray50, linestyle=:dash, linewidth=1.5, label="Total Loss (+ Reg)")
axislegend(ax1, position=:rt)

# Subplot 2: Reaction Force Curve
ax2 = Axis(fig[1, 2], xlabel="Load Step", ylabel="Reaction Force (N)", title="Reaction Force: FE vs PGD-NN")
lines!(ax2, 1:n_steps_total, F_ext_history, label="FE Experimental (All Steps)", color=:black, linewidth=2)
lines!(ax2, 1:n_steps_total, learned_F_all, label="Identified PGD-NN (All Steps)", color=:crimson, linestyle=:dash, linewidth=2)
scatter!(ax2, step_indices, learned_F, label="Identified Points ($n_steps Steps)", color=:crimson, markersize=10)
axislegend(ax2, position=:lt)

plot_save_path = joinpath(save_dir, "euclid_sparse_decomp_results.png")
save(plot_save_path, fig)
mkpath(plotsdir())
save(plotsdir("euclid_sparse_decomp_results.png"), fig)
println("Saved validation plots to: ", plot_save_path)

# Generate Final Architecture Plots (matching Neo_Hook_Model_Architecture_plot.jl)
println("\nGenerating Final Architecture Visualizations...")
nets_x1_final = [models[i].layers[2][1] for i in 1:num_modes]
nets_x2_final = [models[i].layers[2][2] for i in 1:num_modes]
nets_x3_final = [models[i].layers[2][3] for i in 1:num_modes]

fig_overlay_final = plot_individual_nets_overlay(nets_x1_final, nets_x2_final, nets_x3_final, num_modes, raw_scale; title_prefix="Final Model")
save_overlay_path = joinpath(save_dir, "Neo_Hook_Individual_Nets_Overlay.png")
save(save_overlay_path, fig_overlay_final)
save(plotsdir("Visualizations", "Neo_Hook_Individual_Nets_Overlay.png"), fig_overlay_final)
println("Saved Final Individual Nets Overlay to: ", save_overlay_path)

fig_1d_final = plot_1d_subnetworks(models, raw_scale, nets_x1_final, nets_x2_final, nets_x3_final, num_modes; iter="Final")
save_1d_path = joinpath(save_dir, "Neo_Hook_Model_Architecture_1D_Subnetworks.png")
save(save_1d_path, fig_1d_final)
save(plotsdir("Visualizations", "Neo_Hook_Model_Architecture_1D_Subnetworks.png"), fig_1d_final)
display(fig_1d_final)
println("Saved Final 1D Subnetworks Table to: ", save_1d_path)

# Mirror to exp_raw/Final and plots/EUCLID_Noise for downstream evaluations
final_dir = datadir("exp_raw", "Final", "EUCLID_SparseDecomp_Noise")
mkpath(final_dir)
wsave(joinpath(final_dir, "network.jld2"), save_data)
cp(params_txt_path, joinpath(final_dir, "parameters.txt"), force=true)
save(joinpath(final_dir, "Neo_Hook_Individual_Nets_Overlay.png"), fig_overlay_final)
save(joinpath(final_dir, "Neo_Hook_Model_Architecture_1D_Subnetworks.png"), fig_1d_final)
save(joinpath(final_dir, "euclid_sparse_decomp_results.png"), fig)

mkpath(plotsdir("EUCLID_Noise"))
save(plotsdir("EUCLID_Noise", "Model_Architecture_1D_Subnetworks.png"), fig_1d_final)
save(plotsdir("EUCLID_Noise", "Individual_Nets_Overlay.png"), fig_overlay_final)
save(plotsdir("EUCLID_Noise", "Convergence_History.png"), fig)

println("\n" * "="^75)
println("EUCLID Sparse Decomposition Workflow Completed Successfully!")
println("="^75)
