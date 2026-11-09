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

###############################################################################################################
# 1. Reference Strain Energy Function and Analytical Derivatives
###############################################################################################################
const C10 = 0.5f0
const C01 = 0.2f0
const D_VOL = 5.0f0

"""
Reference compressible Neo-Hookean strain energy function:
    Ψ(I₁, I₂, J) = C₁₀ * (I₁ * J^(-2/3) - 3) + C₀₁ * (I₂ * J^(-4/3) - 3) + D * (J - 1)²
"""
function reference_function(x)
    # Invariants: x[1] = I₁, x[2] = I₂, x[3] = J (volume ratio)
    return C10 * (x[1] * x[3]^(-2 / 3) - 3f0) + C01 * (x[2] * x[3]^(-4 / 3) - 3f0) + D_VOL * (x[3] - 1)^2
end

"""
Analytical gradient of reference energy with respect to x = [I₁, I₂, J]:
    [∂Ψ/∂I₁, ∂Ψ/∂I₂, ∂Ψ/∂J]
"""
function grad_psi_ref(x)
    I1, I2, J = x[1], x[2], x[3]
    d_I1 = C10 * J^(-2 / 3)
    d_I2 = C01 * J^(-4 / 3)
    d_J = -(2 / 3) * C10 * I1 * J^(-5 / 3) - (4 / 3) * C01 * I2 * J^(-7 / 3) + 2 * D_VOL * (J - 1)
    return Float32[d_I1, d_I2, d_J]
end

###############################################################################################################
# 2. Load Trained Neural Network from datadir("exp_raw")
###############################################################################################################
function load_latest_network()
    # Check candidate checkpoint directories
    cand_paths = [
        datadir("exp_raw", "Final", "E3_NeoHook", "network.jld2"),
        datadir("exp_raw", "Final", "D3", "network.jld2"),
        datadir("exp_raw", "D3_NeoHook_Split_LBFGS_2026-09-17_11-52-34", "network.jld2"),
    ]
    for c in cand_paths
        if isfile(c)
            println("Loading network from: $c")
            return wload(c), basename(dirname(c))
        end
    end

    # Fallback: scan datadir("exp_raw") for the latest NeoHooke / D3 run
    exp_raw_dir = datadir("exp_raw")
    if isdir(exp_raw_dir)
        subdirs = filter(d -> isdir(joinpath(exp_raw_dir, d)) && isfile(joinpath(exp_raw_dir, d, "network.jld2")), readdir(exp_raw_dir))
        if !isempty(subdirs)
            neohook_dirs = filter(d -> occursin("NeoHook", d) || occursin("D3", d) || occursin("E3", d), subdirs)
            target_dirs = isempty(neohook_dirs) ? subdirs : neohook_dirs
            sort!(target_dirs)
            latest_dir = joinpath(exp_raw_dir, target_dirs[end])
            println("Loading trained model from: $latest_dir")
            return wload(joinpath(latest_dir, "network.jld2")), basename(latest_dir)
        end
    end
    error("No trained network found in $(datadir("exp_raw")). Please run training first (scripts/Supervised/NeoHooke.jl).")
end

model_data, run_name = load_latest_network()
nets_x1 = model_data["nets_x1"]
nets_x2 = model_data["nets_x2"]
nets_x3 = model_data["nets_x3"]
raw_scale = model_data["raw_scale"]
num_modes = model_data["num_modes"]

get_scale(scale_param, m=1) = length(scale_param) == 1 ? (softplus(scale_param[1]) + 1.0f0) : (softplus(scale_param[m]) + 1.0f0)

"""
Neural Network Strain Energy Potential Ψ_NN(x) for x = [I1, I2, J]
"""
function psi_nn(x)
    total = zero(eltype(x))
    for m in 1:num_modes
        S = get_scale(raw_scale, m)
        val = nets_x1[m](x[1:1])[1] + nets_x2[m](x[2:2])[1] + nets_x3[m](x[3:3])[1]
        total += S * (exp(val / S) - 1.0f0)
    end
    return total
end

"""
Gradient of NN energy function with respect to x = [I1, I2, J] via automatic differentiation
"""
function grad_psi_nn(x)
    return Flux.gradient(psi_nn, Float32.(x))[1]
end

###############################################################################################################
# 3. Continuum Mechanics: Stress from Invariants via Chain Rule with respect to C
###############################################################################################################
"""
Compute the Second Piola-Kirchhoff stress tensor S from Right Cauchy-Green tensor C and ∂Ψ/∂x.
Invariants of C:
  I1 = tr(C)
  I2 = 1/2 * (tr(C)^2 - tr(C^2))
  I3 = det(C) = J^2  => J = sqrt(I3)

Derivatives with respect to C:
  ∂I1/∂C = I
  ∂I2/∂C = I1*I - C
  ∂I3/∂C = I3 * C^(-1)

Chain Rule:
  ∂Ψ/∂C  = (∂Ψ/∂I1)*(∂I1/∂C) + (∂Ψ/∂I2)*(∂I2/∂C) + (∂Ψ/∂I3)*(∂I3/∂C)
where ∂Ψ/∂I3 = (∂Ψ/∂J) * (∂J/∂I3) = (1 / (2*J)) * (∂Ψ/∂J).

Second Piola-Kirchhoff Stress:
  S = 2 * ∂Ψ/∂C = 2 * [ (∂Ψ/∂I1)*I + (∂Ψ/∂I2)*(I1*I - C) + (∂Ψ/∂I3)*(I3*C^(-1)) ]
    = 2*(∂Ψ/∂I1)*I + 2*(∂Ψ/∂I2)*(I1*I - C) + J*(∂Ψ/∂J)*C^(-1)
"""
function compute_stress_tensors(F, grad_fn)
    C = F' * F
    I1 = tr(C)
    I2 = 0.5f0 * (I1^2 - tr(C * C))
    I3 = det(C)
    J = sqrt(max(I3, 1f-8))

    x = Float32[I1, I2, J]
    dPsi = grad_fn(x)
    dPsi_dI1 = dPsi[1]
    dPsi_dI2 = dPsi[2]
    dPsi_dJ = dPsi[3]
    dPsi_dI3 = dPsi_dJ / (2.0f0 * J)

    I3x3 = Matrix{Float32}(I, 3, 3)
    invC = inv(C)

    # Second Piola-Kirchhoff stress: S = 2 * ∂Ψ/∂C
    S = 2.0f0 * (dPsi_dI1 .* I3x3 .+ dPsi_dI2 .* (I1 .* I3x3 .- C) .+ dPsi_dI3 .* (I3 .* invC))

    # First Piola-Kirchhoff (nominal/engineering) stress: P = F * S
    P = F * S

    # Cauchy (true) stress: σ = (1/J) * F * S * F'
    sigma = (1.0f0 / J) .* (F * S * F')

    return (S=S, P=P, sigma=sigma, J=J)
end

"""
Robustly find root of f(x) = 0 by expanding bracket outward until sign change,
then bisecting to high precision.
"""
function find_zero_stress(f, x_init; x_min=0.001f0, x_max=20.0f0)
    x0 = clamp(x_init, x_min, x_max)
    delta = 0.05f0 * x0
    a = max(x_min, x0 - delta)
    b = min(x_max, x0 + delta)
    fa = f(a)
    fb = f(b)

    iter = 0
    while fa * fb > 0 && iter < 60
        iter += 1
        factor = 1.25f0^iter
        if abs(fa) < abs(fb)
            a = max(x_min, a - delta * factor)
            fa = f(a)
        else
            b = min(x_max, b + delta * factor)
            fb = f(b)
        end
        if fa * fb > 0 && iter > 10
            a = max(x_min, a * 0.85f0)
            b = min(x_max, b * 1.15f0)
            fa = f(a)
            fb = f(b)
        end
    end

    if fa * fb > 0
        @warn "Could not bracket root for zero lateral stress!" x_init fa fb
        return x_init
    end

    for _ in 1:40
        mid = 0.5f0 * (a + b)
        f_mid = f(mid)
        if abs(f_mid) < 1e-6 || (b - a) < 1e-6
            return mid
        end
        if fa * f_mid < 0
            b = mid
            fb = f_mid
        else
            a = mid
            fa = f_mid
        end
    end
    return 0.5f0 * (a + b)
end

###############################################################################################################
# 4. Homogeneous Deformation Modes (Uniaxial, Pure Shear, Equi-biaxial)
###############################################################################################################
# 1. Uniaxial: lateral stress σ_22 = 0
f_ut_stress(lam, lp, g) = compute_stress_tensors(Float32[lam 0 0; 0 lp 0; 0 0 lp], g).S[2, 2]
f_ut_F(lam, lp) = Float32[lam 0 0; 0 lp 0; 0 0 lp]

# 2. Pure Shear: thickness stress σ_33 = 0
f_ps_stress(lam, l3, g) = compute_stress_tensors(Float32[lam 0 0; 0 1.0f0 0; 0 0 l3], g).S[3, 3]
f_ps_F(lam, l3) = Float32[lam 0 0; 0 1.0f0 0; 0 0 l3]

# 3. Equi-biaxial: thickness stress σ_33 = 0
f_eb_stress(lam, l3, g) = compute_stress_tensors(Float32[lam 0 0; 0 lam 0; 0 0 l3], g).S[3, 3]
f_eb_F(lam, l3) = Float32[lam 0 0; 0 lam 0; 0 0 l3]

"""
Evaluates P_11 continuously across stretch_range by using warm-start continuation
starting at undeformed state (λ = 1.0) and propagating smoothly outward.
"""
function evaluate_mode_continuous(stretch_range, f_stress, f_F, grad_fn)
    idx_one = argmin(abs.(stretch_range .- 1.0f0))
    N = length(stretch_range)
    P11_vals = zeros(Float32, N)

    # Sweep tension forward (1.0 -> 2.0)
    prev_root = 1.0f0
    for i in idx_one:N
        lam = stretch_range[i]
        f_obj(l) = f_stress(lam, l, grad_fn)
        root = find_zero_stress(f_obj, prev_root)
        prev_root = root
        F = f_F(lam, root)
        P11_vals[i] = compute_stress_tensors(F, grad_fn).P[1, 1]
    end

    # Sweep compression backward (1.0 -> 0.8)
    prev_root = 1.0f0
    for i in (idx_one-1):-1:1
        lam = stretch_range[i]
        f_obj(l) = f_stress(lam, l, grad_fn)
        root = find_zero_stress(f_obj, prev_root)
        prev_root = root
        F = f_F(lam, root)
        P11_vals[i] = compute_stress_tensors(F, grad_fn).P[1, 1]
    end

    return P11_vals
end

###############################################################################################################
# 5. Evaluate Loading Curves over Stretch Range
###############################################################################################################
stretch_range = range(0.5f0, 2.0f0, length=120)

println("\nEvaluating deformation modes continuously (Uniaxial, Pure Shear, Equi-biaxial)...")
P11_ut_ref = evaluate_mode_continuous(stretch_range, f_ut_stress, f_ut_F, grad_psi_ref)
P11_ut_nn = evaluate_mode_continuous(stretch_range, f_ut_stress, f_ut_F, grad_psi_nn)

P11_ps_ref = evaluate_mode_continuous(stretch_range, f_ps_stress, f_ps_F, grad_psi_ref)
P11_ps_nn = evaluate_mode_continuous(stretch_range, f_ps_stress, f_ps_F, grad_psi_nn)

P11_eb_ref = evaluate_mode_continuous(stretch_range, f_eb_stress, f_eb_F, grad_psi_ref)
P11_eb_nn = evaluate_mode_continuous(stretch_range, f_eb_stress, f_eb_F, grad_psi_nn)
println("Evaluation complete.")

###############################################################################################################
# 6. Single [1, 1] Plot: P_11 vs Stretch across All Loading Modes
###############################################################################################################
fig = Figure(size=(900, 650), fontsize=15)

ax = Axis(fig[1, 1],
    title="Nominal Stress P₁₁ vs. Stretch λ\nReference vs. Neural Network Model ($run_name)",
    xlabel="Stretch λ [-]",
    ylabel="Nominal Stress P₁₁ [MPa]",
    xgridvisible=true,
    ygridvisible=true
)

# Reference curves (Solid lines)
lines!(ax, stretch_range, P11_ut_ref, color=:crimson, linewidth=2.5, label="Uniaxial (Reference)")
lines!(ax, stretch_range, P11_ps_ref, color=:royalblue, linewidth=2.5, label="Pure Shear (Reference)")
lines!(ax, stretch_range, P11_eb_ref, color=:forestgreen, linewidth=2.5, label="Equi-biaxial (Reference)")

# Neural Network curves (Dashed lines)
lines!(ax, stretch_range, P11_ut_nn, color=:crimson, linestyle=:dash, linewidth=2.5, label="Uniaxial (PGD-ICNN)")
lines!(ax, stretch_range, P11_ps_nn, color=:royalblue, linestyle=:dash, linewidth=2.5, label="Pure Shear (PGD-ICNN)")
lines!(ax, stretch_range, P11_eb_nn, color=:forestgreen, linestyle=:dash, linewidth=2.5, label="Equi-biaxial (PGD-ICNN)")

# Reference lines at undeformed state (λ = 1, P = 0)
vlines!(ax, [1.0f0], color=:gray, linestyle=:dot, linewidth=1.5)
hlines!(ax, [0.0f0], color=:gray, linestyle=:dot, linewidth=1.5)

axislegend(ax, position=:lt, framevisible=true, backgroundcolor=(:white, 0.9))

save_dir_plots = plotsdir("Visualizations")
mkpath(save_dir_plots)
save_plot_path = joinpath(save_dir_plots, "Neo_Hook_Loading_Modes.png")
save(save_plot_path, fig)
printstyled("\nFigure saved successfully to: $save_plot_path\n", color=:green, bold=true)

###############################################################################################################
# 7. 1D Slices of Invariants (I1, I2, J): Ground Truth, Network Prediction, and Noisy Training Data
###############################################################################################################
const X1_BOUNDS = (3.01, 9.0)
const X2_BOUNDS = (3.01, 6.0)
const X3_BOUNDS = (0.5, 3.0)

get_domain_ranges(n) = (
    range(Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]), length=n),
    range(Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]), length=n),
    range(Float32(X3_BOUNDS[1]), Float32(X3_BOUNDS[2]), length=n),
)

function create_grid(n)
    xs1, xs2, xs3 = get_domain_ranges(n)
    n1, n2, n3 = length(xs1), length(xs2), length(xs3)
    x1 = repeat(xs1, inner=n2 * n3)
    x2 = repeat(repeat(xs2, inner=n3), outer=n1)
    x3 = repeat(xs3, outer=n1 * n2)
    return xs1, xs2, xs3, [x1'; x2'; x3']
end

"""
Plot 1D slices across I1, I2, I3 where:
- Analytical reference function Ψ_ref is shown (dashed black line)
- Network prediction Ψ_NN is shown (solid crimson curve)
- Individual noisy training points are shown (scatter points)
- Clean training targets are optionally shown (gray crosses)
"""
function plot_1d_slices(predict_fn::Function, x_train, y_train, reference_function;
    y_train_clean=nothing,
    noise_level=nothing,
    save_path=nothing)
    dim_names = ["I₁", "I₂", "J (I₃)"]
    dim_symbols = ["I₁", "I₂", "J"]
    bounds = [X1_BOUNDS, X2_BOUNDS, X3_BOUNDS]

    u_coords = [unique(sort(round.(x_train[d, :], digits=4))) for d in 1:3]

    fig_slices = Figure(size=(1400, 1100), fontsize=14)
    noise_str = noise_level !== nothing ? " (Noise: $(round(noise_level*100, digits=1))%)" : ""
    Label(fig_slices[0, :], "1D Slices of Strain Energy Potential Ψ(I₁, I₂, J)$noise_str\nModel: $run_name",
        fontsize=20, font=:bold)

    slice_levels = [0.2f0, 0.5f0, 0.8f0]
    level_names = ["Low", "Mid", "High"]

    for d in 1:3
        other_dims = filter(!=(d), 1:3)
        d_a, d_b = other_dims[1], other_dims[2]

        for col in 1:3
            level = slice_levels[col]
            val_a = if length(u_coords[d_a]) >= 3
                idx_a = clamp(round(Int, level * length(u_coords[d_a])), 1, length(u_coords[d_a]))
                u_coords[d_a][idx_a]
            else
                Float32(bounds[d_a][1] + level * (bounds[d_a][2] - bounds[d_a][1]))
            end

            val_b = if length(u_coords[d_b]) >= 3
                idx_b = clamp(round(Int, level * length(u_coords[d_b])), 1, length(u_coords[d_b]))
                u_coords[d_b][idx_b]
            else
                Float32(bounds[d_b][1] + level * (bounds[d_b][2] - bounds[d_b][1]))
            end

            sub_title = "$(level_names[col]) Slice: $(dim_symbols[d_a]) = $(round(val_a, digits=2)), $(dim_symbols[d_b]) = $(round(val_b, digits=2))"
            ax = Axis(fig_slices[d, col],
                title=sub_title,
                xlabel="$(dim_names[d]) [-]",
                ylabel="Ψ [MPa]",
                xgridvisible=true,
                ygridvisible=true)

            xs_dense = range(Float32(bounds[d][1]), Float32(bounds[d][2]), length=150)
            n_dense = length(xs_dense)
            x_eval = Matrix{Float32}(undef, 3, n_dense)
            x_eval[d, :] = collect(xs_dense)
            x_eval[d_a, :] .= val_a
            x_eval[d_b, :] .= val_b

            y_ref = vec(reference_function.(eachcol(x_eval)))
            y_pred = vec(predict_fn(x_eval))

            lines!(ax, xs_dense, y_ref, color=:black, linestyle=:dash, linewidth=2.2, label="Analytical Ψ")
            lines!(ax, xs_dense, y_pred, color=:crimson, linewidth=2.5, label="Network Ψ_NN")

            tol_a = length(u_coords[d_a]) < size(x_train, 2) / 2 ? 1.0f-3 : Float32(0.08 * (bounds[d_a][2] - bounds[d_a][1]))
            tol_b = length(u_coords[d_b]) < size(x_train, 2) / 2 ? 1.0f-3 : Float32(0.08 * (bounds[d_b][2] - bounds[d_b][1]))

            mask = (abs.(x_train[d_a, :] .- val_a) .<= tol_a) .& (abs.(x_train[d_b, :] .- val_b) .<= tol_b)
            n_pts = count(mask)

            if n_pts > 0
                x_pts = x_train[d, mask]
                y_pts = vec(y_train)[mask]

                if y_train_clean !== nothing
                    y_clean_pts = vec(y_train_clean)[mask]
                    scatter!(ax, x_pts, y_clean_pts, marker=:cross, markersize=12,
                        color=:gray40, label="Clean Target")
                end

                scatter!(ax, x_pts, y_pts, marker=:circle, markersize=10,
                    color=(:dodgerblue2, 0.85), strokewidth=1.2, strokecolor=:navy,
                    label="Noisy Train Data ($n_pts pts)")
            end

            if d == 1 && col == 1
                axislegend(ax, position=:lt, framevisible=true, backgroundcolor=(:white, 0.85))
            end
        end
    end

    if save_path !== nothing
        mkpath(dirname(save_path))
        save(save_path, fig_slices)
    end

    return fig_slices
end

# Generate training grid and evaluate 1D slices
println("\nGenerating 1D slices of strain energy potential Ψ(I₁, I₂, J)...")
num_pts_dim = 10
_, _, _, x_tr = create_grid(num_pts_dim)
y_tr_clean = Float32.(vec(reference_function.(eachcol(x_tr))))'
noise_lvl = 0.1f0
Random.seed!(42)
tr_noise = noise_lvl .* randn(Float32, size(y_tr_clean))
y_tr_noisy = max.(1.0f-4, y_tr_clean .* (1.0f0 .+ tr_noise))

predict_psi_batch(x_mat) = Float32[psi_nn(x_mat[:, i]) for i in 1:size(x_mat, 2)]

save_1d_path = joinpath(save_dir_plots, "Neo_Hook_1D_Slices.png")
fig_1d = plot_1d_slices(predict_psi_batch, x_tr, y_tr_noisy, reference_function;
    y_train_clean=y_tr_clean, noise_level=noise_lvl, save_path=save_1d_path)
printstyled("1D Slices figure saved successfully to: $save_1d_path\n", color=:green, bold=true)

display(fig)
display(fig_1d)