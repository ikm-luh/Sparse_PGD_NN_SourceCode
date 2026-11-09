# ==============================================================================
# Supervised Constitutive Identification: Incompressible Polynomial Model
# ------------------------------------------------------------------------------
# Identifies an interpretable, sparse strain energy potential for incompressible
# hyperelasticity:
#     Ψ(I₁, I₂) = C₁₀ * (I₁ - 3)² + C₀₂ * (I₂ - 3)
# using a Proper Generalized Decomposition (PGD) neural network representation:
#     Ψ(I₁, I₂) = ∑ₘ Sₘ [ exp( (f_{m,1}(I₁) + f_{m,2}(I₂)) / Sₘ ) - exp( ... ) ]
#
# Key Features:
#   - 1D Input Convex Neural Network (ICNN) subnets ensuring global convexity (W₂ >= 0, W₃ >= 0).
#   - Rank-1 mode decomposition with L1 and cross-term intra-mode sparsity penalties.
#   - Two-stage hybrid optimization: Adam (exploration + proximal projection)
#     followed by Reparameterized L-BFGS (smooth fine-tuning).
#   - Automatic mode deduplication and inactive mode pruning.
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
Random.seed!(356431)

Makie.inline!(true)

"""
Reference polynomial strain energy function for incompressible hyperelasticity:
    Ψ(I₁, I₂) = C₁₀ * (I₁ - 3)² + C₀₂ * (I₂ - 3)
where C₁₀ = 0.5 MPa and C₀₂ = 0.2 MPa.
"""
function reference_function(x)
    C10 = 0.5f0
    C02 = 0.2f0
    # Incompressible invariants: x[1] = I1, x[2] = I2 (J = 1 identically)
    return C10 * (x[1] - 3.0f0)^2 + C02 * (x[2] - 3.0f0)
end

"""
Analytical gradient of reference energy with respect to x = [I₁, I₂]:
    ∂Ψ/∂I₁ = 2 * C₁₀ * (I₁ - 3)
    ∂Ψ/∂I₂ = C₀₂
"""
function grad_psi_ref(x)
    C10 = 0.5f0
    C02 = 0.2f0
    I1, I2 = x[1], x[2]
    d_I1 = 2.0f0 * C10 * (I1 - 3.0f0)
    d_I2 = C02
    return Float32[d_I1, d_I2]
end

"""
Discriminant condition for real positive principal stretches in incompressible elasticity:
det(F) = 1  <=>  lambda^3 - I1*lambda^2 + I2*lambda - 1 = 0 has 3 positive real roots.
"""
function is_admissible(I1, I2)
    delta = -4f0 * I1^3 - 4f0 * I2^3 + I1^2 * I2^2 + 18f0 * I1 * I2 - 27f0
    return delta >= -1f-3 && I1 >= 3.0f0 && I2 >= 3.0f0
end

"""
Generate training/testing datasets from homogeneous isochoric deformation modes:
1. Uniaxial Tension / Compression (UT): isochoric (det(F) = 1) clamped to evaluation domain
2. Pure Shear / Planar Tension (PS): isochoric (det(F) = 1) clamped to evaluation domain
3. Equi-biaxial Tension / Compression (EB): isochoric (det(F) = 1) clamped to evaluation domain
4. General Biaxial Grid: systematic 2D sweep across stretch space
5. Admissible Iso-I1 and Iso-I2 exploration: decoupled orthogonal sampling within the admissible domain
"""
function generate_special_deformation_data(n_per_mode=300; random_jitter=false, return_info=false)
    pts = Vector{Float32}[]
    stretches = Float32[]
    modes = Symbol[]
    add_jitter(r) = random_jitter ? r .+ 0.005f0 .* randn(Float32, length(r)) : r

    # 1. Uniaxial (UT) - Isochoric (clamped within I1 <= 8, I2 <= 10)
    stretches_ut = add_jitter(range(0.6f0, 3.0f0, length=n_per_mode))
    for lam in stretches_ut
        lp = 1.0f0 / sqrt(max(lam, 0.05f0))
        I1 = lam^2 + 2.0f0 * lp^2
        I2 = lp^4 + 2.0f0 * lam^2 * lp^2
        #if I1 <= 8.2f0 && I2 <= 10.2f0
        push!(pts, Float32[I1, I2])
        push!(stretches, lam)
        push!(modes, :UT)
        #end
    end

    # 2. Pure Shear (PS) - Isochoric (I1 = I2)
    stretches_ps = add_jitter(range(0.6f0, 2.5f0, length=n_per_mode))
    for lam in stretches_ps
        l3 = 1.0f0 / max(lam, 0.05f0)
        I1 = lam^2 + 1.0f0 + l3^2
        I2 = lam^2 + lam^2 * l3^2 + l3^2
        #if I1 <= 8.2f0 && I2 <= 10.2f0
        push!(pts, Float32[I1, I2])
        push!(stretches, lam)
        push!(modes, :PS)
        #end
    end

    # 3. Equi-biaxial (EB) - Isochoric (clamped within domain)
    stretches_eb = add_jitter(range(0.6f0, 2.0f0, length=n_per_mode))
    for lam in stretches_eb
        l3 = 1.0f0 / max(lam^2, 0.01f0)
        I1 = 2.0f0 * lam^2 + l3^2
        I2 = lam^4 + 2.0f0 * lam^2 * l3^2
        #if I1 <= 8.2f0 && I2 <= 10.2f0
        push!(pts, Float32[I1, I2])
        push!(stretches, lam)
        push!(modes, :EB)
        #end
    end

    x_mat = reduce(hcat, pts)
    if return_info
        return x_mat, (stretches=stretches, modes=modes)
    else
        return x_mat
    end
end

"""
Generate test dataset covering the total possible (admissible) space of I1 and I2 data:
I1 in [3.0, 10.0), I2 in [3.0, 16.0), satisfying the incompressibility admissibility condition (delta >= 0).
Supports either uniform random rejection sampling (default) or regular grid sampling.
"""
function generate_admissible_test_data(n_points=1000; i1_min=3.0f0, i1_max=10.0f0, i2_min=3.0f0, i2_max=16.0f0, grid=false, n_grid=40)
    pts = Vector{Float32}[]
    if grid
        r1 = range(i1_min, i1_max - 0.01f0, length=n_grid)
        r2 = range(i2_min, i2_max - 0.01f0, length=n_grid)
        for i1 in r1, i2 in r2
            if is_admissible(i1, i2)
                push!(pts, Float32[i1, i2])
            end
        end
    else
        while length(pts) < n_points
            cand_i1 = i1_min + rand(Float32) * (i1_max - i1_min)
            cand_i2 = i2_min + rand(Float32) * (i2_max - i2_min)
            if cand_i1 < i1_max && cand_i2 < i2_max && is_admissible(cand_i1, cand_i2)
                push!(pts, Float32[cand_i1, cand_i2])
            end
        end
    end
    return reduce(hcat, pts)
end


# Bounds used for slicing/visualizations
const X1_BOUNDS = (3.01, 8.0)
const X2_BOUNDS = (3.01, 10.0)

get_domain_ranges(num_points_per_dim) = (
    range(Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]), length=num_points_per_dim),
    range(Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]), length=num_points_per_dim),
)

split_input(x) = (x[1:1, :], x[2:2, :])
# Reference state in invariant space (undeformed state: I₁ = 3, I₂ = 3 for J = 1)
const X_REF = Float32[3.0; 3.0;;]

"""
Get positive scaling parameter S > 1.0 from unconstrained parameter raw_scale via softplus.
S controls the exponential transition scale in the PGD representation:
    Ψ(x) = ∑ₘ Sₘ * [ exp(Φₘ(x) / Sₘ) - exp(Φₘ(X_REF) / Sₘ) ]
"""
get_scale(scale_param, m=1) = length(scale_param) == 1 ? (softplus(scale_param[1]) + 1.0f0) : (softplus(scale_param[m]) + 1.0f0)

"""
Evaluate total strain energy potential across all modes for input batch x (2 x N).
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
        end for k in 1:2
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

function compute_residual(models, raw_scale, num_points_per_dim; relative=true, exclude_mode=nothing, x_tr=nothing, x_te=nothing)
    x_train_eval = x_tr #!== nothing ? x_tr : generate_special_deformation_data(200; random_jitter=false)
    x_test_eval = x_te #!== nothing ? x_te : generate_special_deformation_data(100; random_jitter=true)

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
        Dense(1 => 20, softplus),
        Dense(20 => 1),
        Dense(1 => 1, bias=false)
    )

    nets_x1 = [net_template() for _ in 1:num_modes]
    nets_x2 = [net_template() for _ in 1:num_modes]

    # Initialize Layer 2 and Layer 3 weights to be non-negative so networks start in convex domain
    # and scaled so that initial exponential predictions stay well-conditioned
    for i in 1:num_modes
        for net in (nets_x1[i], nets_x2[i])
            net.layers[2].weight .= abs.(net.layers[2].weight)
            net.layers[2].weight ./= norm(net.layers[2].weight)
            net.layers[3].weight .= abs.(net.layers[3].weight) .* 0.01f0
        end
        # Break initial symmetry: assign distinct coordinate affinity to the first modes
        if i == 1
            nets_x1[i].layers[3].weight .= 0.05f0
        elseif i == 2
            nets_x2[i].layers[3].weight .= 0.05f0
        elseif i == 3
            nets_x1[i].layers[3].weight .= 0.05f0
        elseif i == 4
            nets_x2[i].layers[3].weight .= 0.05f0
        end
    end

    models = [Chain(split_input, Parallel(+, nets_x1[i], nets_x2[i])) for i in 1:num_modes]

    return nets_x1, nets_x2, models
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

        reg_l1 += w1 + w2
        # Penalizes when both nets are active in the same mode to promote rank-1 separation
        reg_intra += w1 * w2
    end

    return poisson_loss + lam * reg_l1 + lam_cross * reg_intra
end

function train_models_simultaneously!(models, raw_scale, x_train, y_train, opt, eta, lam, lam_cross, n_epochs,
    residual_over_iteration=nothing, x_test=nothing, y_test=nothing)
    println("Initial train loss (Poisson): ", loss_simultaneous(models, raw_scale, x_train, y_train, lam, lam_cross))

    params = (models=models, scale=raw_scale)

    for epoch in 1:n_epochs
        val, grads = Flux.withgradient(p -> loss_simultaneous(p.models, p.scale, x_train, y_train, lam, lam_cross), params)
        Flux.update!(opt, params, grads[1])

        # ICNN projection step: enforce W2 >= 1e-4 and ||W2|| = 1 for all sub-networks of every mode
        # Proximal operator for L1 regularizer combined with non-negativity constraint (W3 >= 0):
        # prox_{lam * eta * ||.||_1 + I_{w >= 0}}(w) = max(0, w - lam * eta)
        for m in models
            for k in 1:2
                w2 = m.layers[2][k][2].weight
                w2 .= max.(1.0f-4, w2)
                n2 = norm(w2)
                if n2 > 1.0f-8
                    w2 ./= n2
                end

                m.layers[2][k][3].weight[1] = max(0.0f0, m.layers[2][k][3].weight[1] - lam * eta)
            end
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
Compute invariants (I₁, I₂) for classical homogeneous deformation modes given principal stretch λ:
- :UT -> Uniaxial: I₁ = λ² + 2/λ, I₂ = 2λ + 1/λ²
- :PS -> Pure Shear: I₁ = I₂ = λ² + 1 + 1/λ²
- :EB -> Equi-biaxial: I₁ = 2λ² + 1/λ⁴, I₂ = λ⁴ + 2/λ²
"""
function deformation_invariants(mode::Symbol, lam)
    if mode == :UT
        lp = 1.0f0 / sqrt(max(lam, 0.05f0))
        I1 = lam^2 + 2.0f0 * lp^2
        I2 = lp^4 + 2.0f0 * lam^2 * lp^2
        return Float32[I1, I2]
    elseif mode == :PS
        l3 = 1.0f0 / max(lam, 0.05f0)
        I1 = lam^2 + 1.0f0 + l3^2
        I2 = lam^2 + lam^2 * l3^2 + l3^2
        return Float32[I1, I2]
    elseif mode == :EB
        l3 = 1.0f0 / max(lam^2, 0.01f0)
        I1 = 2.0f0 * lam^2 + l3^2
        I2 = lam^4 + 2.0f0 * lam^2 * l3^2
        return Float32[I1, I2]
    else
        error("Unknown deformation mode: $mode")
    end
end

"""
Plot the 3 classical homogeneous deformation modes (Uniaxial, Pure Shear, Equi-biaxial)
comparing the learned neural network prediction Ψ_NN against the analytical reference model Ψ
and the noisy training observations:

- Panel 1: Uniaxial Tension/Compression (UT)
- Panel 2: Pure Shear / Planar Tension (PS)
- Panel 3: Equi-biaxial Tension/Compression (EB)
"""
function plot_deformation_modes(predict_fn::Function, x_train, y_train, reference_function;
    train_info=nothing,
    y_train_clean=nothing,
    noise_level=nothing,
    iter=nothing,
    save_path=nothing)
    mode_configs = [
        (:UT, "Uniaxial Mode (UT)", "I₁ = λ² + 2/λ,  I₂ = 2λ + 1/λ²", (0.6f0, 2.75f0)),
        (:PS, "Pure Shear Mode (PS)", "I₁ = I₂ = λ² + 1 + 1/λ²", (0.5f0, 2.6f0)),
        (:EB, "Equi-biaxial Mode (EB)", "I₁ = 2λ² + 1/λ⁴,  I₂ = λ⁴ + 2/λ²", (0.65f0, 1.75f0)),
    ]

    fig = Figure(size=(1600, 540), fontsize=14)
    noise_str = noise_level !== nothing ? " (Noise: $(round(noise_level*100, digits=1))%)" : ""
    iter_str = iter !== nothing ? " [Iter $iter]" : ""
    Label(fig[0, :], "Deformation Test Fits: Analytical vs Neural Network vs Noisy Data$noise_str$iter_str",
        fontsize=20, font=:bold)

    for (col, (mode_sym, mode_title, mode_sub, default_range)) in enumerate(mode_configs)
        ax = Axis(fig[1, col],
            title="$mode_title\n$mode_sub",
            xlabel="Principal Stretch λ [-]",
            ylabel="Strain Energy Potential Ψ [MPa]",
            xgridvisible=true,
            ygridvisible=true)

        # Determine stretch domain from training data if available, else use default range
        lam_min, lam_max = default_range
        if train_info !== nothing
            m_mask = train_info.modes .== mode_sym
            if any(m_mask)
                lam_min = minimum(train_info.stretches[m_mask])
                lam_max = maximum(train_info.stretches[m_mask])
            end
        end

        lams_dense = range(Float32(lam_min), Float32(lam_max), length=200)
        x_dense = Matrix{Float32}(undef, 2, length(lams_dense))
        for (j, lam) in enumerate(lams_dense)
            x_dense[:, j] = deformation_invariants(mode_sym, lam)
        end

        y_ref = vec(reference_function.(eachcol(x_dense)))
        y_pred = vec(predict_fn(x_dense))

        # Ground truth analytical and learned neural network curves
        lines!(ax, lams_dense, y_ref, color=:black, linestyle=:dash, linewidth=2.4, label="Analytical Ψ")
        lines!(ax, lams_dense, y_pred, color=:crimson, linewidth=2.5, label="Network Ψ_NN")

        # Reference line for undeformed state λ = 1.0
        vlines!(ax, [1.0f0], color=(:gray60, 0.7), linestyle=:dot, linewidth=1.2)

        # Plot training data points for this mode
        if train_info !== nothing
            mask = train_info.modes .== mode_sym
            n_pts = count(mask)
            if n_pts > 0
                lams_pts = train_info.stretches[mask]
                y_pts = vec(y_train)[mask]

                #= if y_train_clean !== nothing
                    y_clean_pts = vec(y_train_clean)[mask]
                    scatter!(ax, lams_pts, y_clean_pts, marker=:cross, markersize=10,
                             color=:gray35, label="Clean Target")
                end =#

                scatter!(ax, lams_pts, y_pts, marker=:circle, markersize=8,
                    color=(:dodgerblue2, 0.85), strokewidth=1.0, strokecolor=:navy,
                    label="Noisy Train Data ($n_pts pts)")
            end
        end

        if col == 1
            axislegend(ax, position=:lt, framevisible=true, backgroundcolor=(:white, 0.85))
        end
    end

    for col in 1:3
        colsize!(fig.layout, col, Relative(1 / 3))
    end

    if save_path !== nothing
        mkpath(dirname(save_path))
        save(save_path, fig)
    end

    return fig
end

plot_deformation_modes(models, raw_scale, x_train, y_train, reference_function; kwargs...) =
    plot_deformation_modes(x -> predict_all(models, raw_scale, x), x_train, y_train, reference_function; kwargs...)

function visualize_results(models, raw_scale, reference_function, nets_x1, nets_x2, num_modes, iteration=nothing; iter=nothing, x_eval=nothing)
    current_iter = iter !== nothing ? iter : iteration
    iter_label = current_iter !== nothing ? " (Iteration $current_iter)" : ""

    ###############################################################################################################
    # 3D scatter plots of 2D invariant domain (x1=I1, x2=I2, z=y)
    xs1_2d, xs2_2d = get_domain_ranges(20)
    x1_2d = repeat(xs1_2d, inner=length(xs2_2d))
    x2_2d = repeat(xs2_2d, outer=length(xs1_2d))
    x_2d = [x1_2d'; x2_2d']

    y_true = vec(reference_function.(eachcol(x_2d)))
    y_modes = [
        let S = get_scale(raw_scale, m)
            vec(S .* (exp.(models[m](x_2d) ./ S) .- exp.(models[m](X_REF) ./ S)))
        end for m in 1:num_modes
    ]

    fig_3d = Figure(title="Mode Analysis - 3D Scatter (Domain I₁, I₂)$iter_label", size=(max(1600, 380 * (num_modes + 1)), 1200))
    Label(fig_3d[0, :], "Mode Analysis - 3D Scatter (Domain I₁, I₂)$iter_label", fontsize=20, font=:bold)

    ax1 = Axis3(fig_3d[1, 1], xlabel="I₁", ylabel="I₂", zlabel="Ψ")
    scatter!(ax1, x1_2d, x2_2d, y_true, markersize=5, color=y_true, colormap=:viridis)
    ax1.title = "True Function"

    y_accum = zeros(Float32, length(y_true))
    sum_y_modes = zeros(Float32, length(y_true))

    for m in 1:num_modes
        col = m + 1
        y_accum .+= y_modes[m]
        sum_y_modes .+= y_modes[m]

        y_combined = y_accum
        y_residual = y_true .- y_combined

        ax_pred = Axis3(fig_3d[1, col], xlabel="I₁", ylabel="I₂", zlabel="Ψ")
        scatter!(ax_pred, x1_2d, x2_2d, y_modes[m], markersize=5, color=y_modes[m], colormap=:viridis)
        ax_pred.title = "Mode $m Prediction"

        ax_res = Axis3(fig_3d[2, col], xlabel="I₁", ylabel="I₂", zlabel="residual")
        scatter!(ax_res, x1_2d, x2_2d, y_residual, markersize=5, color=y_residual, colormap=:viridis)
        residual_label = m == 1 ? "True - Mode 1" : "True - " * join(["Mode $k" for k in 1:m], " - ")
        ax_res.title = "Residual ($residual_label)"

        if m >= 2
            ax_comb = Axis3(fig_3d[3, col], xlabel="I₁", ylabel="I₂", zlabel="Ψ")
            scatter!(ax_comb, x1_2d, x2_2d, y_combined, markersize=5, color=y_true .- sum_y_modes, colormap=:viridis)
            combined_label = join(["Mode $k" for k in 1:m], " + ")
            ax_comb.title = "Combined $combined_label"
        end
    end

    #display(fig_3d)

    ###############################################################################################################
    # Net1 / Net2 outputs per mode
    xs1_net, xs2_net = get_domain_ranges(200)
    x1_net = collect(xs1_net)'
    x2_net = collect(xs2_net)'

    f_nets = Figure(size=(max(1000, 280 * num_modes), 680))
    Label(f_nets[0, :], "Sub-network outputs per mode$iter_label", fontsize=16, font=:bold)

    eval_pts = x_eval !== nothing ? x_eval : (@isdefined(x_test) ? x_test : x_2d)
    mode_preds_global = [
        let S = get_scale(raw_scale, k)
            vec(S .* (exp.(models[k](eval_pts) ./ S) .- exp.(models[k](X_REF) ./ S)))
        end for k in 1:num_modes
    ]
    mode_norms_global = [norm(mp, 2) for mp in mode_preds_global]
    sum_norms_global = sum(mode_norms_global)

    ref_x1 = Float32[3.0f0;;]
    ref_x2 = Float32[3.0f0;;]

    for mode in 1:num_modes
        S_mode = get_scale(raw_scale, mode)
        net1_out = vec(S_mode .* (exp.(nets_x1[mode](x1_net) ./ S_mode) .- exp.(nets_x1[mode](ref_x1) ./ S_mode)))
        net2_out = vec(S_mode .* (exp.(nets_x2[mode](x2_net) ./ S_mode) .- exp.(nets_x2[mode](ref_x2) ./ S_mode)))

        ax1 = Axis(f_nets[1, mode],
            title="Mode $mode – Net x1",
            xlabel="I₁",
            ylabel="output")
        lines!(ax1, xs1_net, net1_out, color=:blue, label="Net x1")
        axislegend(ax1, position=:lt)

        ax2 = Axis(f_nets[2, mode],
            title="Mode $mode – Net x2",
            xlabel="I₂",
            ylabel="output")
        lines!(ax2, xs2_net, net2_out, color=:orange, label="Net x2")
        axislegend(ax2, position=:rt)

        mode_share_global = mode_norms_global[mode] / max(sum_norms_global, 1f-8)
        Label(f_nets[3, mode], "scale = $(round(S_mode, digits=2))\nMode Share = $(round(mode_share_global * 100, digits=2))%", fontsize=13, font=:bold)
    end

    rowsize!(f_nets.layout, 3, Fixed(45))

    for col in 1:num_modes
        colsize!(f_nets.layout, col, Relative(1 / num_modes))
    end

    display(f_nets)
end

num_modes = 5
eta = 1f-3 # learning rate
lam = 0.001f0 # initial regularization strength
lam_decay = 1.0f0 # decay factor for regularization strength per iteration
lam_cross_init = 0.0f0 # initial cross penalty strength (warm-up)
lam_cross_max = 0.001f0 # target penalty strength if more than 1 net is active per mode
n_epochs = 500
n_data_per_experiment = 50
num_iterations = 60
iter_max_lam = 10 # iteration at which maximum penalty lambda (lam_cross) is reached
dedup_start_iteration = 20 #20 # iteration at which mode deduplication and combination begins
lbfgs_switch_iteration = 50 # iteration after which optimization switches to L-BFGS
lbfgs_iters_per_checkpoint = n_epochs # match Adam frequency (n_epochs steps per checkpoint)
min_remaining_modes = 1 # minimum number of modes to retain during pruning
share_thresh = 0.001f0   # mode share threshold for removing inactive modes
noise_level = 0.0f0 # additive Gaussian noise scale relative to std(y_train_clean) (e.g. 0.05 = 5% of signal standard deviation)
initial_num_modes = num_modes

nets_x1, nets_x2, models = init_model(num_modes)

# Initial learnable scale S (parameterized as S = softplus(raw_scale[1]) + 1.0f0, initialized at S = 10.0f0)
init_S = 5.0f0
init_raw = Float32(log(exp(init_S) - 1.0f0))
raw_scale = Float32[init_raw]
residual_over_iteration = Float32[]
test_residual_over_iteration = Float32[]

# Training and testing datasets from homogeneous isochoric deformation modes
# (Uniaxial tension/compression, Pure shear, Equi-biaxial, General biaxial)
x_train, train_info = generate_special_deformation_data(n_data_per_experiment; random_jitter=false, return_info=true)
y_train_clean = Float32.(vec(reference_function.(eachcol(x_train))))'
sigma_noise = Float32(noise_level * std(y_train_clean))
y_train = max.(1f-4, y_train_clean .+ sigma_noise .* randn(Float32, size(y_train_clean)))
if noise_level > 0
    println("Added additive Gaussian noise: sigma = $(round(sigma_noise, digits=4)) ($(round(noise_level * 100, digits=1))% of std(y)).")
end

# Test dataset covering the total admissible space of I1 and I2 data with I1 < 10 and I2 < 16
x_test = generate_admissible_test_data(1000; i1_min=3.0f0, i1_max=10.0f0, i2_min=3.0f0, i2_max=16.0f0)
y_test = Float32.(vec(reference_function.(eachcol(x_test))))' # clean reference for testing generalization
println("Generated test dataset on total admissible space: $(size(x_test, 2)) points (I1 < 10, I2 < 16).")

params = (models=models, scale=raw_scale)
global opt = Flux.setup(Flux.Adam(eta), params)

# Initial state before training
global_residual, test_residual = compute_residual(models, raw_scale, n_data_per_experiment; x_tr=x_train, x_te=x_test)
printstyled("Initial global residual: ", global_residual, "\n", color=:blue, bold=true)
printstyled("Initial test residual: ", test_residual, "\n", color=:blue, bold=true)
push!(residual_over_iteration, global_residual)
push!(test_residual_over_iteration, test_residual)
visualize_results(models, raw_scale, reference_function, nets_x1, nets_x2, num_modes, 0; x_eval=x_test)

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

            # ICNN projection step: enforce W2 >= 0 and unit-norm ||W2|| = 1 for sub-networks of every mode
            # Proximal operator for L1 regularizer combined with non-negativity constraint (W3 >= 0):
            # prox_{lam * eta * ||.||_1 + I_{w >= 0}}(w) = max(0, w - lam * eta)
            for m in models
                for k in 1:2
                    w2 = m.layers[2][k][2].weight
                    w2 .= max.(0.0f0, w2)
                    n2 = norm(w2)
                    if n2 > 1.0f-8
                        w2 ./= n2
                    end

                    m.layers[2][k][3].weight[1] = max(0.0f0, m.layers[2][k][3].weight[1] - current_lam * eta)
                end
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
                for k in 1:2
                    subnet = models_v[m].layers[2][k]
                    subnet.layers[2].weight .= inv_sp(subnet.layers[2].weight)
                    w3_val = models[m].layers[2][k].layers[3].weight[1]
                    if w3_val > 0.001f0
                        subnet.layers[3].weight .= inv_sp(subnet.layers[3].weight)
                    else
                        subnet.layers[3].weight .= -20.0f0
                    end
                end
            end

            # Build gradient mask: 1.0 for trainable weights, 0.0 for frozen (pruned)
            frozen_mask_models = deepcopy(models_v)
            for m in 1:num_modes
                for k in 1:2
                    subnet = frozen_mask_models[m].layers[2][k]
                    subnet.layers[1].weight .= 1.0f0
                    subnet.layers[1].bias .= 1.0f0
                    subnet.layers[2].weight .= 1.0f0
                    subnet.layers[2].bias .= 1.0f0
                    w3_val = models[m].layers[2][k].layers[3].weight[1]
                    if w3_val > 0.001f0
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
                for k in 1:2
                    sub_v = models_v[m].layers[2][k]
                    sub_m = models[m].layers[2][k]
                    sub_m.layers[1].weight .= sub_v.layers[1].weight
                    sub_m.layers[1].bias .= sub_v.layers[1].bias
                    w2_pos = softplus.(sub_v.layers[2].weight)
                    sub_m.layers[2].weight .= w2_pos ./ (norm(w2_pos) + 1.0f-8)
                    sub_m.layers[2].bias .= sub_v.layers[2].bias
                    if sub_v.layers[3].weight[1] > -15.0f0
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

            # Keep nets_x1, nets_x2 synced
            for i in 1:num_modes
                nets_x1[i] = models[i].layers[2][1]
                nets_x2[i] = models[i].layers[2][2]
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
    if iter_num <= lbfgs_switch_iteration && iter_num >= dedup_start_iteration && num_modes > min_remaining_modes
        net_lists = [nets_x1, nets_x2]
        net_names = ["x1", "x2"]

        xs1_eval, xs2_eval = get_domain_ranges(200)
        x_evals = [collect(xs1_eval)', collect(xs2_eval)']

        sim_found = false
        target_remove = 0
        survivor = 0
        best_active_net_str = ""

        ref_evals = [Float32[3.0f0;;], Float32[3.0f0;;]]
        for m1 in 1:num_modes
            S_m1 = get_scale(raw_scale, m1)
            sub_outs1 = [vec(S_m1 .* (exp.(net_lists[k][m1](x_evals[k]) ./ S_m1) .- exp.(net_lists[k][m1](ref_evals[k]) ./ S_m1))) for k in 1:2]
            stds1 = [std(so) for so in sub_outs1]
            w_thresh = 0.005f0
            std_thresh = 0.01f0
            active1 = [net_lists[k][m1].layers[3].weight[1] > w_thresh && stds1[k] > std_thresh for k in 1:2]
            if !any(active1)
                continue
            end

            for m2 in (m1+1):num_modes
                S_m2 = get_scale(raw_scale, m2)
                sub_outs2 = [vec(S_m2 .* (exp.(net_lists[l][m2](x_evals[l]) ./ S_m2) .- exp.(net_lists[l][m2](ref_evals[l]) ./ S_m2))) for l in 1:2]
                stds2 = [std(so) for so in sub_outs2]
                active2 = [net_lists[l][m2].layers[3].weight[1] > w_thresh && stds2[l] > std_thresh for l in 1:2]

                # Directly merge if EXACTLY the same sub-networks are active
                if active1 == active2 && any(active1)
                    if mode_norms[m1] >= mode_norms[m2]
                        survivor = m1
                        target_remove = m2
                    else
                        survivor = m2
                        target_remove = m1
                    end
                    best_active_net_str = join(net_names[active1], ", ")
                    share_survivor = round(mode_norms[survivor] / max(sum_norms, 1f-8) * 100, digits=2)
                    share_remove = round(mode_norms[target_remove] / max(sum_norms, 1f-8) * 100, digits=2)
                    printstyled("Modes $m1 and $m2 have the exact same active nets [$best_active_net_str]! Keeping Mode $survivor values (share: $share_survivor%) and removing duplicate Mode $target_remove (share: $share_remove%).\n", color=:yellow, bold=true)
                    sim_found = true
                    break
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
            global num_modes -= 1
            global params = (models=models, scale=raw_scale)
            global opt = Flux.setup(Flux.Adam(eta), params)
            mode_removed_this_iter = true
            printstyled("Total number of modes lowered to $num_modes.\n", color=:yellow, bold=true)
        elseif !sim_found && num_modes > min_remaining_modes
            # Inactive / low energy share mode pruning (removes idle modes with < 2% share or no active subnets)
            for j in 1:num_modes
                mode_share = mode_norms[j] / max(sum_norms, 1f-8)
                sub_outs_j = [vec(get_scale(raw_scale, j) .* (exp.(net_lists[k][j](x_evals[k]) ./ get_scale(raw_scale, j)) .- exp.(net_lists[k][j](ref_evals[k]) ./ get_scale(raw_scale, j)))) for k in 1:2]
                stds_j = [std(so) for so in sub_outs_j]
                w3_j = [net_lists[k][j].layers[3].weight[1] for k in 1:2]
                act_j = [w3_j[k] > 0.005f0 && stds_j[k] > 0.01f0 for k in 1:2]

                if (!any(act_j) || mode_share < share_thresh) && num_modes > min_remaining_modes
                    reason = !any(act_j) ? "inactive subnets" : "share $(round(mode_share * 100, digits=2))% < $(round(share_thresh * 100, digits=1))%"
                    printstyled("Mode $j has $reason. Removing idle mode!\n", color=:magenta, bold=true)
                    deleteat!(models, j)
                    deleteat!(nets_x1, j)
                    deleteat!(nets_x2, j)
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

    global global_residual, test_residual = compute_residual(models, raw_scale, n_data_per_experiment; x_tr=x_train, x_te=x_test)
    printstyled("Global residual: ", global_residual, "\n", color=:blue, bold=true)
    printstyled("Test residual: ", test_residual, "\n", color=:blue, bold=true)

    visualize_results(models, raw_scale, reference_function, nets_x1, nets_x2, num_modes, iter_num; x_eval=x_test)
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
    println("Mode $m:")
    println("  Net x1 -> min(W2) = $(round(w2_x1, digits=5)) (>=0: $(w2_x1 >= 0)), ||W2|| = $(round(norm_w2_x1, digits=5)), W3 = $(round(w3_x1, digits=5)) (>=0: $(w3_x1 >= 0))")
    println("  Net x2 -> min(W2) = $(round(w2_x2, digits=5)) (>=0: $(w2_x2 >= 0)), ||W2|| = $(round(norm_w2_x2, digits=5)), W3 = $(round(w3_x2, digits=5)) (>=0: $(w3_x2 >= 0))")
end
println("All sub-networks satisfy W2 >= 0, ||W2|| ≈ 1, and W3 >= 0 with softplus activations.")
println("=> All 1D sub-networks are guaranteed convex, ensuring global convexity of Psi.")
println("="^60)

###############################################################################################################
# Save network and training results to datadir/exp_raw/<filename>
###############################################################################################################
script_name = "Polynomial"
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
save_dir = datadir("exp_raw", "$(script_name)_$(timestamp)")
mkpath(save_dir)

save_data = Dict(
    "models" => models,
    "nets_x1" => nets_x1,
    "nets_x2" => nets_x2,
    "raw_scale" => raw_scale,
    "model_state" => Flux.state(models),
    "num_modes" => num_modes,
    "residual_over_iteration" => residual_over_iteration,
    "test_residual_over_iteration" => test_residual_over_iteration,
    "global_residual" => global_residual,
    "test_residual" => test_residual,
    "x_train" => x_train,
    "y_train" => y_train,
    "y_train_clean" => y_train_clean,
    "train_info" => train_info,
    "noise_level" => noise_level,
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

# Save hyperparameters and experiment parameters to parameters.txt
params_txt_path = joinpath(save_dir, "parameters.txt")
open(params_txt_path, "w") do io
    println(io, "============================================================")
    println(io, "Training Hyperparameters & Experiment Configuration")
    println(io, "Timestamp: $timestamp")
    println(io, "Script: $script_name")
    println(io, "============================================================")
    println(io, "num_modes                  = $initial_num_modes")
    println(io, "eta                        = $eta")
    println(io, "lam                        = $lam")
    println(io, "lam_decay                  = $lam_decay")
    println(io, "lam_cross_init             = $lam_cross_init")
    println(io, "lam_cross_max              = $lam_cross_max")
    println(io, "n_epochs                   = $n_epochs")
    println(io, "n_data_per_experiment      = $n_data_per_experiment")
    println(io, "num_iterations             = $num_iterations")
    println(io, "iter_max_lam               = $iter_max_lam")
    println(io, "dedup_start_iteration      = $dedup_start_iteration")
    println(io, "lbfgs_switch_iteration     = $lbfgs_switch_iteration")
    println(io, "lbfgs_iters_per_checkpoint = $lbfgs_iters_per_checkpoint")
    println(io, "min_remaining_modes        = $min_remaining_modes")
    println(io, "share_thresh               = $share_thresh")
    println(io, "noise_level                = $noise_level")
    println(io, "init_S                     = $init_S")
    println(io, "------------------------------------------------------------")
    println(io, "Results Summary:")
    println(io, "final_num_modes            = $num_modes")
    println(io, "learned_scale_S            = $(round(get_scale(raw_scale, 1), digits=4))")
    println(io, "global_residual            = $global_residual")
    println(io, "test_residual              = $test_residual")
    println(io, "============================================================")
end
printstyled("Training parameters text document saved to: $params_txt_path\n", color=:green, bold=true)

# Save deformation modes test visualization (UT, PS, EB: Analytical vs Network vs Noisy Data)
save_plot_path = joinpath(save_dir, "Incompressible_Deformation_Modes.png")
fig_modes_final = plot_deformation_modes(models, raw_scale, x_train, y_train, reference_function;
    train_info=train_info,
    y_train_clean=y_train_clean, noise_level=noise_level,
    save_path=save_plot_path)
save_viz_path = joinpath(plotsdir("Visualizations"), "Incompressible_Deformation_Modes.png")
mkpath(dirname(save_viz_path))
save(save_viz_path, fig_modes_final)
printstyled("Deformation modes figure saved successfully to: $save_plot_path\n", color=:green, bold=true)
display(fig_modes_final)