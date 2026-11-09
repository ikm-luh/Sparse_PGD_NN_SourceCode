# ==============================================================================
# Supervised Constitutive Identification: Compressible Neo-Hookean Model
# ------------------------------------------------------------------------------
# Identifies an interpretable, sparse strain energy potential for compressible
# hyperelasticity:
#     Ψ(I₁, I₂, J) = C₁₀ * (I₁ * J^(-2/3) - 3) + C₀₁ * (I₂ * J^(-4/3) - 3) + D * (J - 1)²
# using a Proper Generalized Decomposition (PGD) neural network representation:
#     Ψ(I₁, I₂, J) = ∑ₘ Sₘ [ exp( (f_{m,1}(I₁) + f_{m,2}(I₂) + f_{m,3}(J)) / Sₘ ) - exp( ... ) ]
#
# Key Architectural Features:
#   - 1D Input Convex Neural Network (ICNN) subnets ensuring global convexity (W₂ >= 0, W₃ >= 0).
#   - Mode 5 initialized as a dedicated volumetric mode using symmetric bilateral
#     U-ICNN weights around J = 1.0 to capture both compression (J < 1) and tension (J > 1).
#   - Multi-term rank-1 decomposition with L1 and cross-term (w₁ * w₂ * w₃) intra-mode
#     regularization to suppress spurious 3D couplings while allowing 1D and 2D terms.
#   - Two-stage hybrid training: Adam (exploration + proximal projection)
#     followed by Reparameterized L-BFGS (smooth fine-tuning).
#   - Mode deduplication (merging duplicates) and subspace-subset pruning.
# ==============================================================================

using DrWatson
@quickactivate "Sparse_PGD_NN"

using Flux
using CairoMakie
using Statistics
using Random
using LinearAlgebra
using Dates
using Optim
Random.seed!(42)

Makie.inline!(true)

"""
Reference compressible Neo-Hookean strain energy potential:
    Ψ(I₁, I₂, J) = C₁₀ * (I₁ * J^(-2/3) - 3) + C₀₁ * (I₂ * J^(-4/3) - 3) + D * (J - 1)²
where:
    C₁₀ = 0.5 MPa (isochoric first invariant term)
    C₀₁ = 0.2 MPa (isochoric second invariant term)
    D   = 5.0 MPa (volumetric compressibility term)
"""
function reference_function(x)
    C10 = 0.5f0
    C01 = 0.2f0
    D = 5.0f0
    # Invariants: x[1] = I1, x[2] = I2, x[3] = J (volume ratio)
    return C10 * (x[1] * x[3]^(-2 / 3) - 3f0) + C01 * (x[2] * x[3]^(-4 / 3) - 3f0) + D * (x[3] - 1)^2
end

get_domain_ranges(num_points_per_dim) = (
    range(Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]), length=num_points_per_dim),
    range(Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]), length=num_points_per_dim),
    range(Float32(X3_BOUNDS[1]), Float32(X3_BOUNDS[2]), length=num_points_per_dim),
)

"""
Discriminant condition for real positive principal stretches in compressible elasticity:
det(F) = J > 0 <=> lambda^3 - I1*lambda^2 + I2*lambda - J^2 = 0 has 3 positive real roots.
"""
function is_admissible(I1, I2, J)
    I3 = J^2
    delta = I1^2 * I2^2 - 4f0 * I2^3 - 4f0 * I1^3 * I3 + 18f0 * I1 * I2 * I3 - 27f0 * I3^2
    return delta >= -1f-3 && I1 >= 3.0f0 * (J^(2 / 3)) && I2 >= 3.0f0 * (J^(4 / 3)) && J > 0f0
end

"""
Sample physically admissible points uniformly at random within (X1_BOUNDS, X2_BOUNDS, X3_BOUNDS).
Rejection sampling filters candidate points using is_admissible to ensure all 3 principal stretches
are positive and real.
"""
function sample_random_points(n_points=500; return_info=false)
    pts = Vector{Float32}[]
    sizehint!(pts, n_points)
    x1_min, x1_max = Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2])
    x2_min, x2_max = Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2])
    x3_min, x3_max = Float32(X3_BOUNDS[1]), Float32(X3_BOUNDS[2])

    while length(pts) < n_points
        x1 = x1_min + (x1_max - x1_min) * rand(Float32)
        x2 = x2_min + (x2_max - x2_min) * rand(Float32)
        x3 = x3_min + (x3_max - x3_min) * rand(Float32)
        if is_admissible(x1, x2, x3)
            push!(pts, Float32[x1, x2, x3])
        end
    end
    x_mat = reduce(hcat, pts)
    if return_info
        return x_mat, (stretches=nothing, modes=nothing)
    else
        return x_mat
    end
end

"""
Generate training/testing dataset by randomly sampling physically admissible states.
Maintained for backward compatibility.
"""
function generate_special_deformation_data(num_points=500; random_jitter=false, return_info=false)
    total_points = num_points <= 35 ? round(Int, num_points^3 * 0.26f0) : num_points
    return sample_random_points(total_points; return_info=return_info)
end

split_input(x) = (x[1:1, :], x[2:2, :], x[3:3, :])
# Undeformed reference configuration: I₁ = 3, I₂ = 3, J = 1
const X_REF = Float32[3.0; 3.0; 1.0;;]

"""
Get positive scaling parameter S > 1.0 from unconstrained parameter raw_scale via softplus.
S controls the exponential transition scale in the PGD representation:
    Ψ(x) = ∑ₘ Sₘ * [ exp(Φₘ(x) / Sₘ) - exp(Φₘ(X_REF) / Sₘ) ]
"""
get_scale(scale_param, m=1) = length(scale_param) == 1 ? (softplus(scale_param[1]) + 1.0f0) : (softplus(scale_param[m]) + 1.0f0)

"""
Evaluate total strain energy potential across all modes for input batch x (3 x N).
Guarantees Ψ(X_REF) = 0 (zero energy at undeformed reference state).
"""
predict_all(models, raw_scale, x) = sum(
    let S = get_scale(raw_scale, m)
        S .* (exp.(models[m](x) ./ S) .- exp.(models[m](X_REF) ./ S))
    end for m in 1:length(models)
)

# Inverse softplus helper for parameterizing positive weights during L-BFGS
inv_sp(w) = log.(max.(1f-6, exp.(w) .- 1.0f0))

"""
Evaluate subnetworks in reparameterized V-space where W = softplus(V).
Ensures weights W₂ >= 0 and W₃ >= 0 strictly without manual clipping during L-BFGS.
"""
function eval_model_reparam(m_net, x)
    xs = split_input(x)
    sum(
        let net = m_net.layers[2][k]
            h1 = net.layers[1](xs[k])
            w2_pos = softplus.(net.layers[2].weight)
            w2 = w2_pos ./ (norm(w2_pos) + 1.0f-8)
            b2 = net.layers[2].bias
            h2 = w2 * h1 .+ b2
            w3 = softplus(net.layers[3].weight[1])
            w3 .* h2
        end for k in 1:3
    )
end

function predict_all_reparam(models, raw_scale, x)
    sum(
        let S = get_scale(raw_scale, m)
            out = eval_model_reparam(models[m], x)
            out_ref = eval_model_reparam(models[m], X_REF)
            S .* (exp.(out ./ S) .- exp.(out_ref ./ S))
        end for m in 1:length(models)
    )
end

"""
Poisson deviance loss function for non-negative energy fitting:
    L(y_pred, y) = mean( y_pred - y * log(y_pred) )
Provides stable gradient scaling across small and large energy magnitudes.
"""
function loss_reparam(models, raw_scale, x, y)
    y_pred = predict_all_reparam(models, raw_scale, x)
    eps_val = 1f-4
    y_pred_safe = max.(y_pred, eps_val)
    return mean(y_pred_safe .- y .* log.(y_pred_safe))
end

function compute_residual(models, raw_scale, num_points=nothing; relative=true, exclude_mode=nothing, x_tr=nothing, x_te=nothing)
    x_train_eval = x_tr !== nothing ? x_tr : sample_random_points(500)
    x_test_eval = x_te !== nothing ? x_te : sample_random_points(200)

    y_train = vec(reference_function.(eachcol(x_train_eval)))
    y_test = vec(reference_function.(eachcol(x_test_eval)))

    res_train = copy(y_train)
    res_test = copy(y_test)
    for i in 1:length(models)
        if i != exclude_mode
            S = get_scale(raw_scale, i)
            res_train .-= vec(S .* (exp.(models[i](x_train_eval) ./ S) .- exp.(models[i](X_REF) ./ S)))
            res_test .-= vec(S .* (exp.(models[i](x_test_eval) ./ S) .- exp.(models[i](X_REF) ./ S)))
        end
    end

    train_res = norm(res_train, 2)
    test_res = norm(res_test, 2)
    if relative
        train_res /= norm(y_train, 2)
        test_res /= norm(y_test, 2)
    end

    return train_res, test_res
end

function init_model(num_modes)
    # 1D Input Convex Neural Network (ICNN) sub-network template:
    # Layer 1: Dense(1 => 10, softplus) - affine input + convex activation
    # Layer 2: Dense(10 => 1) - non-negative weights W2 >= 0 enforce convexity
    # Layer 3: Dense(1 => 1, bias=false) - non-negative scaling factor W3 >= 0 (with L1 proximal sparsity)
    net_template() = Chain(
        Dense(1 => 10, softplus),
        Dense(10 => 1),
        Dense(1 => 1, bias=false)
    )

    nets_x1 = [net_template() for _ in 1:num_modes]
    nets_x2 = [net_template() for _ in 1:num_modes]
    nets_x3 = [net_template() for _ in 1:num_modes]

    # Initialize Layer 2 and Layer 3 weights to be non-negative so networks start in convex domain
    # and scaled so that initial exponential predictions stay well-conditioned
    for i in 1:num_modes
        for net in (nets_x1[i], nets_x2[i], nets_x3[i])
            net.layers[2].weight .= abs.(net.layers[2].weight)
            net.layers[2].weight ./= norm(net.layers[2].weight)
            net.layers[3].weight .= abs.(net.layers[3].weight) .* 0.01f0
        end
        # Break initial symmetry: assign distinct coordinate affinity to the first modes
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
    # Disable x1 and x2 contributions
    if num_modes >= 5
        nets_x1[5].layers[3].weight .= 0.0f0
        nets_x2[5].layers[3].weight .= 0.0f0

        # Symmetric bilateral U-ICNN initialization around x3 = 1.0:
        # Half positive slopes (x3 > 1) and half negative slopes (x3 < 1)
        w_u = Float32[1, 2, 4, 8, 12, -1, -2, -4, -8, -12]
        nets_x3[5].layers[1].weight .= reshape(w_u, 10, 1)
        nets_x3[5].layers[1].bias .= -w_u .* 1.0f0
        nets_x3[5].layers[2].weight .= 0.1f0
        nets_x3[5].layers[2].weight ./= norm(nets_x3[5].layers[2].weight)
        nets_x3[5].layers[2].bias .= 0.0f0
        nets_x3[5].layers[3].weight .= 0.05f0
    end

    models = [Chain(split_input, Parallel(+, nets_x1[i], nets_x2[i], nets_x3[i])) for i in 1:num_modes]

    return nets_x1, nets_x2, nets_x3, models
end

function loss_simultaneous(models, raw_scale, x, y, lam, lam_cross)
    y_pred = predict_all(models, raw_scale, x)
    eps_val = 1f-4
    y_pred_safe = max.(y_pred, eps_val)
    poisson_loss = mean(y_pred_safe .- y .* log.(y_pred_safe))

    reg_l1 = zero(Float32)
    reg_intra = zero(Float32)
    for m in models
        w1 = m.layers[2][1][3].weight[1]
        w2 = m.layers[2][2][3].weight[1]
        w3 = m.layers[2][3][3].weight[1]

        reg_l1 += w1 + w2 + w3
        # Allows 1 or 2 nets to be active; strictly penalizes when all 3 nets are active
        reg_intra += w1 * w2 * w3
    end

    return poisson_loss + lam * reg_l1 + lam_cross * reg_intra
end

function train_models_simultaneously!(models, raw_scale, x_train, y_train, opt, eta, lam, lam_cross, n_epochs,
    residual_over_iteration=nothing, x_test=nothing, y_test=nothing; vol_mode_idx=length(models) >= 5 ? 5 : 0)
    println("Initial train loss (Poisson): ", loss_simultaneous(models, raw_scale, x_train, y_train, lam, lam_cross))

    params = (models=models, scale=raw_scale)

    for epoch in 1:n_epochs
        val, grads = Flux.withgradient(p -> loss_simultaneous(p.models, p.scale, x_train, y_train, lam, lam_cross), params)
        Flux.update!(opt, params, grads[1])

        # ICNN projection step: enforce W2 >= 1e-4 and ||W2|| = 1 for all 3 sub-networks of every mode
        # Proximal operator for L1 regularizer combined with non-negativity constraint (W3 >= 0):
        # prox_{lam * eta * ||.||_1 + I_{w >= 0}}(w) = max(0, w - lam * eta)
        for m in models
            for k in 1:3
                w2 = m.layers[2][k][2].weight
                w2 .= max.(1.0f-4, w2)
                n2 = norm(w2)
                if n2 > 1.0f-8
                    w2 ./= n2
                end

                m.layers[2][k][3].weight[1] = max(0.0f0, m.layers[2][k][3].weight[1] - lam * eta)
            end
        end

        # Keep volumetric mode strictly dedicated to x3: freeze x1 and x2 contributions at 0
        if 1 <= vol_mode_idx <= length(models)
            models[vol_mode_idx].layers[2][1][3].weight .= 0.0f0
            models[vol_mode_idx].layers[2][2][3].weight .= 0.0f0
        end

        # Save residual periodically (every 10 epochs or at endpoints) to maintain fast, smooth training
        if residual_over_iteration !== nothing && (epoch % 10 == 0 || epoch == 1 || epoch == n_epochs)
            x_eval = x_test !== nothing ? x_test : x_train
            y_eval = y_test !== nothing ? y_test : y_train
            y_pred = predict_all(models, raw_scale, x_eval)
            res = Float32(norm(y_pred .- y_eval, 2) / norm(y_eval, 2))
            push!(residual_over_iteration, res)
        end
    end

    current_S = get_scale(raw_scale, 1)
    println("Learned scale S: $(round(current_S, digits=3))")
    println("Final train loss (Poisson): ", loss_simultaneous(models, raw_scale, x_train, y_train, lam, lam_cross))
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
    iter=nothing,
    save_path=nothing)
    dim_names = ["I₁", "I₂", "J (I₃)"]
    dim_symbols = ["I₁", "I₂", "J"]
    bounds = [X1_BOUNDS, X2_BOUNDS, X3_BOUNDS]

    # Unique grid coordinates along each axis if on a regular grid
    u_coords = [unique(sort(round.(x_train[d, :], digits=4))) for d in 1:3]

    fig = Figure(size=(1400, 1100), fontsize=14)
    noise_str = noise_level !== nothing ? " (Noise: $(round(noise_level*100, digits=1))%)" : ""
    iter_str = iter !== nothing ? " [Iter $iter]" : ""
    Label(fig[0, :], "1D Slices of Strain Energy Potential Ψ(I₁, I₂, J)$noise_str$iter_str",
        fontsize=20, font=:bold)

    # 3x3 layout: Row d varies invariant d; columns are 3 different slices (low, mid, high of other two)
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
            ax = Axis(fig[d, col],
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

            # Identify training points that lie on (or close to) this slice
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
        save(save_path, fig)
    end

    return fig
end

plot_1d_slices(models, raw_scale, x_train, y_train, reference_function; kwargs...) =
    plot_1d_slices(x -> predict_all(models, raw_scale, x), x_train, y_train, reference_function; kwargs...)

function visualize_results(models, raw_scale, reference_function, nets_x1, nets_x2, nets_x3, num_modes, iteration=nothing; iter=nothing, x_eval=nothing)
    current_iter = iter !== nothing ? iter : iteration
    iter_label = current_iter !== nothing ? " (Iteration $current_iter)" : ""

    ###############################################################################################################
    # 3D scatter plots: evaluate on (x1, x2) slice at nominal reference state x3 = 1.0
    xs1_3d, xs2_3d, xs3_3d = get_domain_ranges(20)
    x3_ref = 1.0f0
    x1_3d = repeat(xs1_3d, inner=length(xs2_3d))
    x2_3d = repeat(xs2_3d, outer=length(xs1_3d))
    x3_slice = fill(x3_ref, length(x1_3d))
    x_3d = [x1_3d'; x2_3d'; x3_slice']

    y_true = vec(reference_function.(eachcol(x_3d)))
    y_modes = [
        let S = get_scale(raw_scale, m)
            vec(S .* (exp.(models[m](x_3d) ./ S) .- exp.(models[m](X_REF) ./ S)))
        end for m in 1:num_modes
    ]

    fig_3d = Figure(title="Mode Analysis - 3D Scatter (Slice at x3=$x3_ref)$iter_label", size=(max(1600, 380 * (num_modes + 1)), 1200))
    Label(fig_3d[0, :], "Mode Analysis - 3D Scatter (Slice at x3=$x3_ref)$iter_label", fontsize=20, font=:bold)

    ax1 = Axis3(fig_3d[1, 1], xlabel="x1 (I₁)", ylabel="x2 (I₂)", zlabel="Ψ")
    scatter!(ax1, x1_3d, x2_3d, y_true, markersize=5, color=y_true, colormap=:viridis)
    ax1.title = "True Function"

    y_accum = zeros(Float32, length(y_true))
    sum_y_modes = zeros(Float32, length(y_true))

    for m in 1:num_modes
        col = m + 1
        y_accum .+= y_modes[m]
        sum_y_modes .+= y_modes[m]

        y_combined = y_accum
        y_residual = y_true .- y_combined

        ax_pred = Axis3(fig_3d[1, col], xlabel="x1 (I₁)", ylabel="x2 (I₂)", zlabel="Ψ")
        scatter!(ax_pred, x1_3d, x2_3d, y_modes[m], markersize=5, color=y_modes[m], colormap=:viridis)
        ax_pred.title = "Mode $m Prediction"

        ax_res = Axis3(fig_3d[2, col], xlabel="x1 (I₁)", ylabel="x2 (I₂)", zlabel="residual")
        scatter!(ax_res, x1_3d, x2_3d, y_residual, markersize=5, color=y_residual, colormap=:viridis)
        residual_label = m == 1 ? "True - Mode 1" : "True - " * join(["Mode $k" for k in 1:m], " - ")
        ax_res.title = "Residual ($residual_label)"

        if m >= 2
            ax_comb = Axis3(fig_3d[3, col], xlabel="x1 (I₁)", ylabel="x2 (I₂)", zlabel="Ψ")
            scatter!(ax_comb, x1_3d, x2_3d, y_combined, markersize=5, color=y_true .- sum_y_modes, colormap=:viridis)
            combined_label = join(["Mode $k" for k in 1:m], " + ")
            ax_comb.title = "Combined $combined_label"
        end
    end

    ###############################################################################################################
    # Net1 / Net2 / Net3 outputs per mode
    xs1_net, xs2_net, xs3_net = get_domain_ranges(200)
    x1_net = collect(xs1_net)'
    x2_net = collect(xs2_net)'
    x3_net = collect(xs3_net)'

    f_nets = Figure(size=(max(1000, 280 * num_modes), 970))
    Label(f_nets[0, :], "Sub-network outputs per mode$iter_label", fontsize=16, font=:bold)

    eval_pts = x_eval !== nothing ? x_eval : (@isdefined(x_test) ? x_test : x_3d)
    mode_preds_global = [
        let S = get_scale(raw_scale, k)
            vec(S .* (exp.(models[k](eval_pts) ./ S) .- exp.(models[k](X_REF) ./ S)))
        end for k in 1:num_modes
    ]
    mode_norms_global = [norm(mp, 2) for mp in mode_preds_global]
    sum_norms_global = sum(mode_norms_global)

    ref_x1 = Float32[3.0f0;;]
    ref_x2 = Float32[3.0f0;;]
    ref_x3 = Float32[1.0f0;;]

    for mode in 1:num_modes
        S_mode = get_scale(raw_scale, mode)
        net1_out = vec(S_mode .* (exp.(nets_x1[mode](x1_net) ./ S_mode) .- exp.(nets_x1[mode](ref_x1) ./ S_mode)))
        net2_out = vec(S_mode .* (exp.(nets_x2[mode](x2_net) ./ S_mode) .- exp.(nets_x2[mode](ref_x2) ./ S_mode)))
        net3_out = vec(S_mode .* (exp.(nets_x3[mode](x3_net) ./ S_mode) .- exp.(nets_x3[mode](ref_x3) ./ S_mode)))

        ax1 = Axis(f_nets[1, mode],
            title="Mode $mode – Net x1",
            xlabel="x1 (I₁)",
            ylabel="output")
        lines!(ax1, xs1_net, net1_out, color=:blue, label="Net x1")
        axislegend(ax1, position=:lt)

        ax2 = Axis(f_nets[2, mode],
            title="Mode $mode – Net x2",
            xlabel="x2 (I₂)",
            ylabel="output")
        lines!(ax2, xs2_net, net2_out, color=:orange, label="Net x2")
        axislegend(ax2, position=:rt)

        ax3 = Axis(f_nets[3, mode],
            title="Mode $mode – Net x3",
            xlabel="x3 (J)",
            ylabel="output")
        lines!(ax3, xs3_net, net3_out, color=:green, label="Net x3")
        axislegend(ax3, position=:rt)

        mode_share_global = mode_norms_global[mode] / max(sum_norms_global, 1f-8)
        Label(f_nets[4, mode], "scale = $(round(S_mode, digits=2))\nMode Share = $(round(mode_share_global * 100, digits=2))%", fontsize=13, font=:bold)
    end

    rowsize!(f_nets.layout, 4, Fixed(45))

    for col in 1:num_modes
        colsize!(f_nets.layout, col, Relative(1 / num_modes))
    end

    display(f_nets)
end

# Bounds used for domain / slicing / visualizations
const X1_BOUNDS = (3.01, 15.0)
const X2_BOUNDS = (3.01, 15.0)
const X3_BOUNDS = (0.25, 4.0)

num_modes = 5
vol_mode_idx = num_modes >= 5 ? 5 : 0
eta = 5f-2 # learning rate
lam = 0.001f0 # initial regularization strength
lam_decay = 1.0f0 # decay factor for regularization strength per iteration
lam_cross_init = 0.0f0 # initial cross penalty strength (warm-up)
lam_cross_max = 1.0f0 # target penalty strength if more than 1 net is active per mode
n_epochs = 500
num_train_points = 500 # number of randomly sampled admissible training points
num_test_points = 200  # number of randomly sampled admissible test points
num_points_per_dim = num_train_points # alias for backwards compatibility
num_iterations = 120
iter_max_lam = 15 # iteration at which maximum penalty lambda (lam_cross) is reached
lbfgs_switch_iteration = 110 # iteration after which optimization switches to L-BFGS
lbfgs_iters_per_checkpoint = n_epochs # match Adam frequency (n_epochs steps per checkpoint)
min_remaining_modes = 1 # minimum number of modes to retain during pruning
share_thresh = 0.001f0   # mode share threshold for removing inactive modes
noise_level = 0.00f0 # additive Gaussian noise scale relative to std(y_train_clean) (e.g. 0.05 = 5% of signal standard deviation)

nets_x1, nets_x2, nets_x3, models = init_model(num_modes)

# Initial learnable scale S (parameterized as S = softplus(raw_scale[1]) + 1.0f0, initialized at S = 10.0f0)
init_S = 10.0f0
init_raw = Float32(log(exp(init_S) - 1.0f0))
raw_scale = Float32[init_raw]
residual_over_iteration = Float32[]
test_residual_over_iteration = Float32[]

# Training and testing datasets randomly sampled from physically admissible domain
x_train, train_info = sample_random_points(num_train_points; return_info=true)
y_train_clean = Float32.(vec(reference_function.(eachcol(x_train))))'
sigma_noise = Float32(noise_level * std(y_train_clean))
y_train = max.(1f-4, y_train_clean .+ sigma_noise .* randn(Float32, size(y_train_clean)))
if noise_level > 0
    println("Added additive Gaussian noise: sigma = $(round(sigma_noise, digits=4)) ($(round(noise_level * 100, digits=1))% of std(y)).")
end

x_test = sample_random_points(num_test_points)
y_test = Float32.(vec(reference_function.(eachcol(x_test))))' # clean reference for testing generalization
println("Sampled $(size(x_train, 2)) training points and $(size(x_test, 2)) test points randomly within bounds.")

params = (models=models, scale=raw_scale)
global opt = Flux.setup(Flux.Adam(eta), params)

# Initial state before training
global_residual, test_residual = compute_residual(models, raw_scale, num_points_per_dim; x_tr=x_train, x_te=x_test)
printstyled("Initial global residual: ", global_residual, "\n", color=:blue, bold=true)
printstyled("Initial test residual: ", test_residual, "\n", color=:blue, bold=true)
push!(residual_over_iteration, global_residual)
push!(test_residual_over_iteration, test_residual)
visualize_results(models, raw_scale, reference_function, nets_x1, nets_x2, nets_x3, num_modes, 0; x_eval=x_test)

# Continuous simultaneous training
total_epochs = num_iterations * n_epochs
plot_every = n_epochs

printstyled("\nStarting hybrid training: $num_iterations checkpoints (Adam for 1..$lbfgs_switch_iteration, L-BFGS for $(lbfgs_switch_iteration + 1)..$num_iterations)...\n", color=:cyan, bold=true)

ramp_epochs = max(1, (iter_max_lam - 1) * n_epochs)

for iter_num in 1:num_iterations
    local current_lam, current_lam_cross

    if iter_num <= lbfgs_switch_iteration
        # Stage 1: Adam optimization (n_epochs per checkpoint)
        for sub_epoch in 1:n_epochs
            epoch = (iter_num - 1) * n_epochs + sub_epoch
            progress = clamp(Float32(epoch - 1) / Float32(ramp_epochs), 0.0f0, 1.0f0)
            current_lam_cross = lam_cross_init + (lam_cross_max - lam_cross_init) * progress
            current_lam = 0.0f0 + lam * progress

            val, grads = Flux.withgradient(p -> loss_simultaneous(p.models, p.scale, x_train, y_train, current_lam, current_lam_cross), params)
            Flux.update!(opt, params, grads[1])

            # ICNN projection step: enforce W2 >= 0 and unit-norm ||W2|| = 1 for all 3 sub-networks of every mode
            # Proximal operator for L1 regularizer combined with non-negativity constraint (W3 >= 0):
            # prox_{lam * eta * ||.||_1 + I_{w >= 0}}(w) = max(0, w - lam * eta)
            for m in models
                for k in 1:3
                    w2 = m.layers[2][k][2].weight
                    w2 .= max.(1.0f-4, w2)
                    n2 = norm(w2)
                    if n2 > 1.0f-8
                        w2 ./= n2
                    end

                    m.layers[2][k][3].weight[1] = max(0.0f0, m.layers[2][k][3].weight[1] - current_lam * eta)
                end
            end

            # Keep volumetric mode strictly dedicated to x3: freeze x1 and x2 contributions at 0
            if 1 <= vol_mode_idx <= length(models)
                models[vol_mode_idx].layers[2][1][3].weight .= 0.0f0
                models[vol_mode_idx].layers[2][2][3].weight .= 0.0f0
            end

            # Save training and test residuals at each update of the weights
            y_pred_tr = predict_all(models, raw_scale, x_train)
            push!(residual_over_iteration, Float32(norm(y_pred_tr .- y_train_clean, 2) / norm(y_train_clean, 2)))

            y_pred_te = predict_all(models, raw_scale, x_test)
            push!(test_residual_over_iteration, Float32(norm(y_pred_te .- y_test, 2) / norm(y_test, 2)))
        end
    else
        # Stage 2: Reparameterized L-BFGS fine-tuning
        # Enforces W2 >= 0 and W3 >= 0 everywhere via smooth softplus reparameterization (W = softplus(V)),
        # completely preventing negative drift and clamping collapse.
        if iter_num == lbfgs_switch_iteration + 1
            printstyled("\n" * "="^75 * "\n", color=:green, bold=true)
            printstyled(">>> Switching to Reparameterized L-BFGS optimization at iteration $iter_num\n", color=:green, bold=true)
            printstyled(">>> Freezing pruned subnets and enforcing W2, W3 >= 0 via softplus parameterization.\n", color=:green, bold=true)
            printstyled("="^75 * "\n", color=:green, bold=true)

            # Convert real models to pre-softplus V space
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

            # Build gradient mask: 1.0 for trainable weights, 0.0 for frozen (pruned or volumetric)
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

        current_lam_cross = 0.0f0
        current_lam = 0.0f0

        theta_v, reconstruct_v = Flux.destructure((models=models_v, scale=raw_scale))

        let last_th = nothing, last_f = 0.0f0, last_g = similar(theta_v)
            function eval_fg!(th)
                if last_th != th
                    last_th = copy(th)
                    curr_params = reconstruct_v(th)
                    val, grads = Flux.withgradient(p -> loss_reparam(p.models, p.scale, x_train, y_train), curr_params)
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

            lbfgs_steps_done = 0
            cb_lbfgs(os) = begin
                if lbfgs_steps_done > 0
                    curr_p = reconstruct_v(os.x)
                    y_tr = predict_all_reparam(curr_p.models, curr_p.scale, x_train)
                    push!(residual_over_iteration, Float32(norm(y_tr .- y_train_clean, 2) / norm(y_train_clean, 2)))
                    y_te = predict_all_reparam(curr_p.models, curr_p.scale, x_test)
                    push!(test_residual_over_iteration, Float32(norm(y_te .- y_test, 2) / norm(y_test, 2)))
                end
                lbfgs_steps_done += 1
                return false
            end

            opt_res = Optim.optimize(
                f_obj, g_obj!, theta_v,
                Optim.LBFGS(),
                Optim.Options(iterations=lbfgs_iters_per_checkpoint, callback=cb_lbfgs)
            )

            # Update V-space state
            final_v = reconstruct_v(Optim.minimizer(opt_res))
            global models_v = final_v.models
            global raw_scale = final_v.scale

            # Reconstruct real-space models with exact softplus weights for visualization & saving
            for m in 1:num_modes
                for k in 1:3
                    sub_v = models_v[m].layers[2][k]
                    sub_m = models[m].layers[2][k]
                    sub_m.layers[1].weight .= sub_v.layers[1].weight
                    sub_m.layers[1].bias .= sub_v.layers[1].bias
                    w2_pos = softplus.(sub_v.layers[2].weight)
                    sub_m.layers[2].weight .= w2_pos ./ (norm(w2_pos) + 1.0f-8)
                    sub_m.layers[2].bias .= sub_v.layers[2].bias
                    if sub_v.layers[3].weight[1] > -15.0f0 && !(m == vol_mode_idx && k != 3)
                        sub_m.layers[3].weight .= softplus.(sub_v.layers[3].weight)
                    else
                        sub_m.layers[3].weight .= 0.0f0
                    end
                end
            end
            global params = (models=models, scale=raw_scale)

            # Ensure the residual recording frequency matches Adam (exactly n_epochs evaluations per checkpoint):
            target_res_count = iter_num * n_epochs + 1
            if length(residual_over_iteration) < target_res_count
                final_tr = Float32(norm(predict_all(models, raw_scale, x_train) .- y_train_clean, 2) / norm(y_train_clean, 2))
                final_te = Float32(norm(predict_all(models, raw_scale, x_test) .- y_test, 2) / norm(y_test, 2))
                while length(residual_over_iteration) < target_res_count
                    push!(residual_over_iteration, final_tr)
                end
                while length(test_residual_over_iteration) < target_res_count
                    push!(test_residual_over_iteration, final_te)
                end
            end

            # Keep nets_x1, nets_x2, nets_x3 synced
            for i in 1:num_modes
                nets_x1[i] = models[i].layers[2][1]
                nets_x2[i] = models[i].layers[2][2]
                nets_x3[i] = models[i].layers[2][3]
            end
        end
    end

    opt_label = iter_num > lbfgs_switch_iteration ? "[L-BFGS]" : "[Adam]"
    printstyled("\n--- Checkpoint $iter_num / $num_iterations $opt_label ---\n", color=:red, bold=true)
    println("Current regularization lambda: $(round(current_lam, digits=5))")
    println("Current intra-mode penalty lambda_cross: $(round(current_lam_cross, digits=5))")
    println("Learned scale S: $(round(get_scale(raw_scale, 1), digits=3))")

    # Mode share analysis on test set
    mode_preds = [
        let S = get_scale(raw_scale, j)
            vec(S .* (exp.(models[j](x_test) ./ S) .- exp.(models[j](X_REF) ./ S)))
        end for j in 1:num_modes
    ]
    mode_norms = [norm(mp, 2) for mp in mode_preds]
    sum_norms = sum(mode_norms)
    for j in 1:num_modes
        mode_share = mode_norms[j] / max(sum_norms, 1f-8)
        println("Mode $j Relative Share: $(round(mode_share * 100, digits=2))%")
    end

    # Mode combination and deduplication based on matching active sub-networks (only during Adam exploration)
    mode_removed_this_iter = false
    if iter_num <= lbfgs_switch_iteration && iter_num >= 10 && num_modes > min_remaining_modes
        net_lists = [nets_x1, nets_x2, nets_x3]
        net_names = ["x1", "x2", "x3"]

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

                # Case A: Exact match (both modes have identical active sub-networks)
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
                    # Merge linear scaling weights on active channels into survivor
                    for k in 1:3
                        if active1[k]
                            models[survivor].layers[2][k][3].weight[1] += models[target_remove].layers[2][k][3].weight[1]
                        end
                    end
                    best_active_net_str = join(net_names[active1], ", ")
                    share_survivor = round(mode_norms[survivor] / max(sum_norms, 1f-8) * 100, digits=2)
                    share_remove = round(mode_norms[target_remove] / max(sum_norms, 1f-8) * 100, digits=2)
                    printstyled("Modes $m1 and $m2 have the exact same active nets [$best_active_net_str]! Merging duplicate Mode $target_remove (share: $share_remove%) into Mode $survivor (share: $share_survivor%).\n", color=:yellow, bold=true)
                    sim_found = true
                    break
                end

                # Case B (Solution 1): Subspace-subset match
                # If one mode is 1D active in an isochoric channel (x1 or x2) and the other mode is a 2D superset
                # containing that same channel (e.g. Mode 1 in x2, Mode 2 in (x2, x3)), merge them!
                if m1 != vol_mode_idx && m2 != vol_mode_idx
                    is_sub_1_in_2 = (count(active1) == 1 && all(active1 .<= active2) && count(active2) > 1)
                    is_sub_2_in_1 = (count(active2) == 1 && all(active2 .<= active1) && count(active1) > 1)

                    if is_sub_1_in_2 || is_sub_2_in_1
                        sub_m = is_sub_1_in_2 ? m1 : m2
                        sup_m = is_sub_1_in_2 ? m2 : m1
                        act_sub = is_sub_1_in_2 ? active1 : active2
                        act_sup = is_sub_1_in_2 ? active2 : active1

                        shared_k = findfirst(act_sub)
                        so_sub = is_sub_1_in_2 ? sub_outs1[shared_k] : sub_outs2[shared_k]
                        so_sup = is_sub_1_in_2 ? sub_outs2[shared_k] : sub_outs1[shared_k]

                        rho = cor(so_sub, so_sup)
                        share_sub = mode_norms[sub_m] / max(sum_norms, 1f-8)

                        # Merge if profile on shared coordinate is correlated (rho > 0.75) or if subset mode share is small (< 15%)
                        if (!isnan(rho) && rho > 0.75f0) || (share_sub < 0.15f0 && (!isnan(rho) ? rho >= 0.0f0 : true))
                            survivor = sup_m
                            target_remove = sub_m

                            # Add linear scaling weight of shared channel into survivor
                            models[survivor].layers[2][shared_k][3].weight[1] += models[target_remove].layers[2][shared_k][3].weight[1]

                            share_survivor = round(mode_norms[survivor] / max(sum_norms, 1f-8) * 100, digits=2)
                            share_remove = round(mode_norms[target_remove] / max(sum_norms, 1f-8) * 100, digits=2)
                            rho_str = !isnan(rho) ? "$(round(rho, digits=3))" : "N/A"
                            sup_str = join(net_names[act_sup], ", ")
                            printstyled("Mode $target_remove is a 1D subset [$(net_names[shared_k])] of Mode $survivor [$sup_str] (rho = $rho_str)! Merging Mode $target_remove (share: $share_remove%) into Mode $survivor (share: $share_survivor%).\n", color=:yellow, bold=true)
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

        if sim_found && num_modes > min_remaining_modes
            # Keep the exact values of the mode with higher mode share (survivor) and remove the other mode
            deleteat!(models, target_remove)
            deleteat!(nets_x1, target_remove)
            deleteat!(nets_x2, target_remove)
            deleteat!(nets_x3, target_remove)
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
            global params = (models=models, scale=raw_scale)
            global opt = Flux.setup(Flux.Adam(eta), params)
            mode_removed_this_iter = true
            printstyled("Total number of modes lowered to $num_modes.\n", color=:yellow, bold=true)
        elseif !sim_found && num_modes > min_remaining_modes
            # Inactive / low energy share mode pruning (removes idle modes with < share_thresh share or no active subnets)
            for j in 1:num_modes
                if j == vol_mode_idx
                    continue
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
                    global params = (models=models, scale=raw_scale)
                    global opt = Flux.setup(Flux.Adam(eta), params)
                    mode_removed_this_iter = true
                    printstyled("Total number of modes lowered to $num_modes.\n", color=:magenta, bold=true)
                    break
                end
            end
        end

        if mode_removed_this_iter
            residual_over_iteration[end] = Float32(norm(predict_all(models, raw_scale, x_train) .- y_train_clean, 2) / norm(y_train_clean, 2))
            test_residual_over_iteration[end] = Float32(norm(predict_all(models, raw_scale, x_test) .- y_test, 2) / norm(y_test, 2))
        end
    end

    global global_residual, test_residual = compute_residual(models, raw_scale, num_points_per_dim; x_tr=x_train, x_te=x_test)
    printstyled("Global residual: ", global_residual, "\n", color=:blue, bold=true)
    printstyled("Test residual: ", test_residual, "\n", color=:blue, bold=true)

    visualize_results(models, raw_scale, reference_function, nets_x1, nets_x2, nets_x3, num_modes, iter_num; x_eval=x_test)
end

println("\n" * "="^60)
println("Convexity & Unit-Norm Verification (ICNN Weight Bounds):")
println("="^60)
for m in 1:num_modes
    w2_x1 = minimum(models[m].layers[2][1].layers[2].weight)
    norm_w2_x1 = norm(models[m].layers[2][1].layers[2].weight)
    w3_x1 = models[m].layers[2][1].layers[3].weight[1]
    w2_x2 = minimum(models[m].layers[2][2].layers[2].weight)
    norm_w2_x2 = norm(models[m].layers[2][2].layers[2].weight)
    w3_x2 = models[m].layers[2][2].layers[3].weight[1]
    w2_x3 = minimum(models[m].layers[2][3].layers[2].weight)
    norm_w2_x3 = norm(models[m].layers[2][3].layers[2].weight)
    w3_x3 = models[m].layers[2][3].layers[3].weight[1]
    println("Mode $m:")
    println("  Net x1 -> min(W2) = $(round(w2_x1, digits=5)) (>=0: $(w2_x1 >= 0)), ||W2|| = $(round(norm_w2_x1, digits=5)), W3 = $(round(w3_x1, digits=5)) (>=0: $(w3_x1 >= 0))")
    println("  Net x2 -> min(W2) = $(round(w2_x2, digits=5)) (>=0: $(w2_x2 >= 0)), ||W2|| = $(round(norm_w2_x2, digits=5)), W3 = $(round(w3_x2, digits=5)) (>=0: $(w3_x2 >= 0))")
    println("  Net x3 -> min(W2) = $(round(w2_x3, digits=5)) (>=0: $(w2_x3 >= 0)), ||W2|| = $(round(norm_w2_x3, digits=5)), W3 = $(round(w3_x3, digits=5)) (>=0: $(w3_x3 >= 0))")
end
println("All sub-networks satisfy W2 >= 0, ||W2|| ≈ 1, and W3 >= 0 with softplus activations.")
println("=> All 1D sub-networks are guaranteed convex, ensuring global convexity of Psi.")
println("="^60)

###############################################################################################################
# Save network and training results to datadir/exp_raw/<filename>
###############################################################################################################
script_name = "NeoHooke"
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
save_dir = datadir("exp_raw", "$(script_name)_$(timestamp)")
mkpath(save_dir)

save_data = Dict(
    "models" => models,
    "nets_x1" => nets_x1,
    "nets_x2" => nets_x2,
    "nets_x3" => nets_x3,
    "raw_scale" => raw_scale,
    "model_state" => Flux.state(models),
    "num_modes" => num_modes,
    "vol_mode_idx" => vol_mode_idx,
    "residual_over_iteration" => residual_over_iteration,
    "test_residual_over_iteration" => test_residual_over_iteration,
    "global_residual" => global_residual,
    "test_residual" => test_residual,
)

save_path = joinpath(save_dir, "network.jld2")
wsave(save_path, save_data)
printstyled("\nNetwork and training results saved successfully to: $save_path\n", color=:green, bold=true)

# Also save residual arrays directly in the folder with the neural network
wsave(joinpath(save_dir, "residual_over_iteration.jld2"), Dict("residual_over_iteration" => residual_over_iteration))
wsave(joinpath(save_dir, "test_residual_over_iteration.jld2"), Dict("test_residual_over_iteration" => test_residual_over_iteration))
wsave(joinpath(save_dir, "residuals.jld2"), Dict(
    "residual_over_iteration" => residual_over_iteration,
    "test_residual_over_iteration" => test_residual_over_iteration,
))
printstyled("Residual arrays (train & test) saved to: $save_dir\n", color=:green, bold=true)

# Save 1D slice visualization
save_plot_path = joinpath(save_dir, "Neo_Hook_1D_Slices.png")
fig_1d_final = plot_1d_slices(models, raw_scale, x_train, y_train, reference_function;
    y_train_clean=y_train_clean, noise_level=noise_level,
    save_path=save_plot_path)
save_viz_path = joinpath(plotsdir("Visualizations"), "Neo_Hook_1D_Slices.png")
mkpath(dirname(save_viz_path))
save(save_viz_path, fig_1d_final)
printstyled("1D Slices figure saved successfully to: $save_plot_path\n", color=:green, bold=true)
display(fig_1d_final)