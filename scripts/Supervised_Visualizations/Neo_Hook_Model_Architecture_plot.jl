using DrWatson
@quickactivate "Sparse_PGD_NN"

using Flux
using CairoMakie
using Statistics
using Random
using LinearAlgebra
using Dates
using LaTeXStrings

Random.seed!(42)
Makie.inline!(true)

set_theme!(theme_latexfonts(), fontsize=30, linewidth=4)

###############################################################################################################
# 1. Reference Strain Energy Function (Neo-Hookean model with 3 invariants: I1, I2, J)
###############################################################################################################
const C10 = 0.5f0
const C01 = 0.2f0
const D_VOL = 5.0f0

"""
Reference strain energy function:
Ψ(I1, I2, J) = C10 * (I1 * J^(-2/3) - 3) + C01 * (I2 * J^(-4/3) - 3) + D * (J - 1)^2
"""
function reference_function(x)
    C10 = 0.5f0
    C01 = 0.2f0
    D = 5.0f0
    # Invariants: x[1] = I1, x[2] = I2, x[3] = J (volume ratio)
    return C10 * (x[1] * x[3]^(-2 / 3) - 3f0) + C01 * (x[2] * x[3]^(-4 / 3) - 3f0) + D * (x[3] - 1)^2
end

"""
Analytical gradient of reference energy with respect to x = [I1, I2, J]:
[∂Ψ/∂I1, ∂Ψ/∂I2, ∂Ψ/∂J]
"""
function grad_psi_ref(x)
    C10 = 0.5f0
    C01 = 0.2f0
    D = 5.0f0
    I1, I2, J = x[1], x[2], x[3]
    d_I1 = C10 * J^(-2 / 3f0)
    d_I2 = C01 * J^(-4 / 3f0)
    d_J = -(2 / 3f0) * C10 * I1 * J^(-5 / 3f0) - (4 / 3f0) * C01 * I2 * J^(-7 / 3f0) + 2f0 * D * (J - 1f0)
    return Float32[d_I1, d_I2, d_J]
end

"""
Discriminant condition for real positive principal stretches in compressible elasticity:
det(F) = J > 0 <=> lambda^3 - I1*lambda^2 + I2*lambda - J^2 = 0 has 3 positive real roots.
"""
function is_admissible(I1, I2, J)
    I3 = J^2
    delta = I1^2 * I2^2 - 4f0 * I2^3 - 4f0 * I1^3 * I3 + 18f0 * I1 * I2 * I3 - 27f0 * I3^2
    return delta >= -1f-3 && I1 >= 3.0f0 * (J^(2 / 3)) && I2 >= 3.0f0 * (J^(4 / 3)) && J > 0f0
end

# Domain intervals for x1, x2, and x3
const X1_BOUNDS = (2.01, 20.0)
const X2_BOUNDS = (2.01, 20.0)
const X3_BOUNDS = (0.25, 5.0)

get_domain_ranges(num_points_per_dim) = (
    range(Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]), length=num_points_per_dim),
    range(Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]), length=num_points_per_dim),
    range(Float32(X3_BOUNDS[1]), Float32(X3_BOUNDS[2]), length=num_points_per_dim),
)

# Reference state in invariant space (undeformed reference configuration: I1 = 3, I2 = 3, J = 1)
const X_REF = Float32[3.0; 3.0; 1.0;;]
const REF_X1 = Float32[3.0;;]
const REF_X2 = Float32[3.0;;]
const REF_X3 = Float32[1.0;;]

split_input(x) = (x[1:1, :], x[2:2, :], x[3:3, :])
get_scale(scale_param, m=1) = length(scale_param) == 1 ? (softplus(scale_param[1]) + 1.0f0) : (softplus(scale_param[m]) + 1.0f0)

function predict_all(models, raw_scale, x)
    psi_raw = sum(
        let S = get_scale(raw_scale, m)
            S .* (exp.(models[m](x) ./ S) .- exp.(models[m](X_REF) ./ S))
        end for m in 1:length(models)
    )
    J = x[3:3, :]
    return psi_raw .- p0 .* (J .- 1.0f0)
end

"""
Load trained Neo-Hookean model from Final directory or latest exp_raw run
"""
function load_network()
    final_path = datadir("exp_raw", "Final", "E3_NeoHook", "network.jld2")
    if isfile(final_path)
        println("Loading network from: $final_path")
        return wload(final_path), basename(dirname(final_path))
    end

    exp_raw_dir = datadir("exp_raw")
    subdirs = filter(d -> isdir(joinpath(exp_raw_dir, d)) && isfile(joinpath(exp_raw_dir, d, "network.jld2")), readdir(exp_raw_dir))
    if isempty(subdirs)
        error("No trained network found in $(exp_raw_dir). Please run training first.")
    end
    # Prioritize D3 / NeoHook runs
    d3_dirs = filter(d -> occursin("E3_NeoHook", d) || occursin("NeoHook", d) || occursin("C3_NeoHook", d), subdirs)
    target_dirs = isempty(d3_dirs) ? subdirs : d3_dirs
    sort!(target_dirs)
    latest_dir = joinpath(exp_raw_dir, target_dirs[end])
    println("Loading trained model from: $latest_dir")
    return wload(joinpath(latest_dir, "network.jld2")), basename(latest_dir)
end

model_data, folder_name = load_network()
nets_x1 = model_data["nets_x1"]
nets_x2 = model_data["nets_x2"]
nets_x3 = model_data["nets_x3"]
raw_scale = model_data["raw_scale"]
num_modes = model_data["num_modes"]
residual_over_iteration = get(model_data, "residual_over_iteration", nothing)
test_residual_over_iteration = get(model_data, "test_residual_over_iteration", nothing)

# Ensure Mode 1 is I₁ and Mode 2 is I₂ for consistent visualization
if num_modes >= 2
    ref1_1 = nets_x1[1](REF_X1)[1]
    ref2_1 = nets_x2[1](REF_X2)[1]
    out1_1 = maximum(abs, nets_x1[1](collect(range(Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]), length=20))') .- ref1_1)
    out2_1 = maximum(abs, nets_x2[1](collect(range(Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]), length=20))') .- ref2_1)
    if out2_1 > out1_1
        nets_x1[1], nets_x1[2] = nets_x1[2], nets_x1[1]
        nets_x2[1], nets_x2[2] = nets_x2[2], nets_x2[1]
        nets_x3[1], nets_x3[2] = nets_x3[2], nets_x3[1]
        if length(raw_scale) >= 2
            raw_scale[1], raw_scale[2] = raw_scale[2], raw_scale[1]
        end
    end
end

models = [Chain(split_input, Parallel(+, nets_x1[i], nets_x2[i], nets_x3[i])) for i in 1:num_modes]

function df_dx_subnet(net, x)
    W1 = net.layers[1].weight
    b1 = net.layers[1].bias
    W2 = net.layers[2].weight
    W3 = net.layers[3].weight[1]
    z1 = W1 * x .+ b1
    sig = Flux.sigmoid.(z1)
    return W3 .* (W2 * (sig .* W1))
end

"""
Compute batch gradients [∂Ψ_NN/∂I1, ∂Ψ_NN/∂I2, ∂Ψ_NN/∂J] for x_mat (3 x N)
using the exact analytical derivatives of the 1D ICNN subnetworks.
"""
function grad_psi_nn_batch(models, raw_scale, x_mat)
    N = size(x_mat, 2)
    dPsi_dI1 = zeros(Float32, N)
    dPsi_dI2 = zeros(Float32, N)
    dPsi_dJ = zeros(Float32, N)

    I1_row = x_mat[1:1, :]
    I2_row = x_mat[2:2, :]
    J_row = x_mat[3:3, :]

    for m in 1:length(models)
        S_m = get_scale(raw_scale, m)
        net1 = models[m].layers[2][1]
        net2 = models[m].layers[2][2]
        net3 = models[m].layers[2][3]

        u1 = net1(I1_row)
        u2 = net2(I2_row)
        u3 = net3(J_row)
        u_tot = vec(u1 .+ u2 .+ u3)
        E_m = exp.(u_tot ./ S_m)

        du1 = vec(df_dx_subnet(net1, I1_row))
        du2 = vec(df_dx_subnet(net2, I2_row))
        du3 = vec(df_dx_subnet(net3, J_row))

        dPsi_dI1 .+= E_m .* du1
        dPsi_dI2 .+= E_m .* du2
        dPsi_dJ .+= E_m .* du3
    end

    return dPsi_dI1, dPsi_dI2, dPsi_dJ
end

"""
Compute reference hydrostatic pressure p0 at undeformed reference configuration X_REF = [3.0; 3.0; 1.0]:
p0 = 2(∂Ψ/∂I₁) + 4(∂Ψ/∂I₂) + ∂Ψ/∂J
Subtracting p0 from ∂Ψ/∂J (or - p0*(J-1) in energy) guarantees a stress-free reference configuration (P(I) = 0).
"""
function compute_p0(models, raw_scale)
    if haskey(model_data, "p0")
        return Float32(model_data["p0"])
    end
    dI1_0, dI2_0, dJ_0 = grad_psi_nn_batch(models, raw_scale, X_REF)
    return 2.0f0 * dI1_0[1] + 4.0f0 * dI2_0[1] + dJ_0[1]
end

const p0 = compute_p0(models, raw_scale)

println("Loaded model with $num_modes modes from $folder_name.")
println("Computed reference hydrostatic pressure p0 = $p0")

###############################################################################################################
# 1. 1D Sub-Network Outputs per Mode across X1_BOUNDS, X2_BOUNDS, and X3_BOUNDS
###############################################################################################################
function plot_1d_subnetworks(models, raw_scale, nets_x1, nets_x2, nets_x3, num_modes; n_pts=200)
    xs1_net, xs2_net, xs3_net = get_domain_ranges(n_pts)
    x1_net = collect(xs1_net)'
    x2_net = collect(xs2_net)'
    x3_net = collect(xs3_net)'

    # Evaluate 3D grid for mode share computation
    xs1_grid, xs2_grid, xs3_grid = get_domain_ranges(15)
    n1, n2, n3 = length(xs1_grid), length(xs2_grid), length(xs3_grid)
    x1_3d = repeat(xs1_grid, inner=n2 * n3)
    x2_3d = repeat(repeat(xs2_grid, inner=n3), outer=n1)
    x3_3d = repeat(xs3_grid, outer=n1 * n2)
    x_3d = [x1_3d'; x2_3d'; x3_3d']

    mode_preds_global = [
        let S = get_scale(raw_scale, k)
            vec(S .* (exp.(models[k](x_3d) ./ S) .- exp.(models[k](X_REF) ./ S)))
        end for k in 1:num_modes
    ]
    mode_norms_global = [norm(mp, 2) for mp in mode_preds_global]
    sum_norms_global = sum(mode_norms_global)

    active_modes = num_modes
    total_cols = active_modes

    f_nets = Figure(size=(max(900, 480 * total_cols), 1350))

    for mode in 1:active_modes
        S_mode = get_scale(raw_scale, mode)

        ref1_val = nets_x1[mode](REF_X1)[1]
        ref2_val = nets_x2[mode](REF_X2)[1]
        ref3_val = nets_x3[mode](REF_X3)[1]

        # Raw outputs u(I) before scale transformation (relative to reference state)
        raw1_out = vec(nets_x1[mode](x1_net) .- ref1_val)
        raw2_out = vec(nets_x2[mode](x2_net) .- ref2_val)
        raw3_out = vec(nets_x3[mode](x3_net) .- ref3_val)

        is_act1 = maximum(abs, raw1_out) > 1e-3
        is_act2 = maximum(abs, raw2_out) > 1e-3
        is_act3 = maximum(abs, raw3_out) > 1e-3

        # Scaling factor distribution:
        # If only one sub-network is active, S is only used on this mode.
        # If two are active, sqrt(S) is used on both.
        # If three are active, cbrt(S) is used on all three.
        # Inactive sub-networks get scale 1.0.
        n_act = count([is_act1, is_act2, is_act3])
        scale_act = n_act == 1 ? S_mode : (n_act == 2 ? sqrt(S_mode) : (n_act == 3 ? S_mode^(1.0f0 / 3.0f0) : 1.0f0))
        scale1 = is_act1 ? scale_act : 1.0f0
        scale2 = is_act2 ? scale_act : 1.0f0
        scale3 = is_act3 ? scale_act : 1.0f0

        # Sub-network outputs without energy normalization:
        h1_out = vec(scale1 .* exp.(nets_x1[mode](x1_net) ./ S_mode))
        h2_out = vec(scale2 .* exp.(nets_x2[mode](x2_net) ./ S_mode))
        h3_out = vec(scale3 .* exp.(nets_x3[mode](x3_net) ./ S_mode))

        h1_0 = scale1 * exp(ref1_val / S_mode)
        h2_0 = scale2 * exp(ref2_val / S_mode)
        h3_0 = scale3 * exp(ref3_val / S_mode)

        mode_share_global = mode_norms_global[mode] / max(sum_norms_global, 1f-8)
        share_pct = round(mode_share_global * 100, digits=1)

        # Header: Mode title on top of each column, and mode share directly below
        header_layout = GridLayout(f_nets[1, mode], tellwidth=false)
        Label(header_layout[1, 1], "Mode $mode", fontsize=34, font=:bold)
        Label(header_layout[2, 1], "Share: $share_pct%", fontsize=26, font=:bold)
        rowgap!(header_layout, 5)

        # 1. Net x1 (I1)
        ax1 = Axis(f_nets[2, mode],
            title="Net I₁",
            xlabel="I₁",
            xgridvisible=true,
            ygridvisible=true)
        lines!(ax1, xs1_net, h1_out, color=:blue, linewidth=4.0, label=L"g_1(I_1)")
        vlines!(ax1, [15.0f0], color=:black, linestyle=:dot, linewidth=2.0)
        xlims!(ax1, Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]))
        if !is_act1
            ylims!(ax1, 0.0, 2.0)
        end
        axislegend(ax1, position=:lt, framevisible=false)

        # 2. Net x2 (I2)
        ax2 = Axis(f_nets[3, mode],
            title="Net I₂",
            xlabel="I₂",
            xgridvisible=true,
            ygridvisible=true)
        lines!(ax2, xs2_net, h2_out, color=:orange, linewidth=4.0, label=L"g_2(I_2)")
        vlines!(ax2, [15.0f0], color=:black, linestyle=:dot, linewidth=2.0)
        xlims!(ax2, Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]))
        if !is_act2
            ylims!(ax2, 0.0, 2.0)
        end
        axislegend(ax2, position=:lt, framevisible=false)

        # 3. Net x3 (J)
        ax3 = Axis(f_nets[4, mode],
            title="Net J",
            xlabel="J",
            xgridvisible=true,
            ygridvisible=true)
        vlines!(ax3, [4.0f0], color=:black, linestyle=:dot, linewidth=2.0)
        xlims!(ax3, Float32(X3_BOUNDS[1]), Float32(X3_BOUNDS[2]))

        if !is_act3
            lines!(ax3, xs3_net, h3_out, color=:forestgreen, linewidth=4.0, label=L"g_3(J)")
            ylims!(ax3, 0.0, 2.0)
            axislegend(ax3, position=:lt, framevisible=false)
        elseif is_act1 && !is_act2
            lines!(ax3, xs3_net, h3_out, color=:forestgreen, linewidth=4.0, label=L"g_3(J)")
            j_pow = xs3_net .^ (-2 / 3f0)
            A_fit = [j_pow ones(Float32, length(j_pow))]
            ab_fit = A_fit \ h3_out
            j_ref = ab_fit[1] .* j_pow .+ ab_fit[2]
            lines!(ax3, xs3_net, j_ref, color=:crimson, linestyle=:dot, linewidth=5, label=L"J^{-2/3}")
            axislegend(ax3, position=:rt, framevisible=false)
        elseif !is_act1 && is_act2
            lines!(ax3, xs3_net, h3_out, color=:forestgreen, linewidth=4.0, label=L"g_3(J)")
            j_pow = xs3_net .^ (-4 / 3f0)
            A_fit = [j_pow ones(Float32, length(j_pow))]
            ab_fit = A_fit \ h3_out
            j_ref = ab_fit[1] .* j_pow .+ ab_fit[2]
            lines!(ax3, xs3_net, j_ref, color=:crimson, linestyle=:dot, linewidth=5, label=L"J^{-4/3}")
            axislegend(ax3, position=:rt, framevisible=false)
        elseif !is_act1 && !is_act2 && is_act3
            # Dedicated volumetric mode D*(J-1)^2 with pressure correction
            h3_phys = h3_out .- p0 .* (xs3_net .- 1.0f0)
            lines!(ax3, xs3_net, h3_phys, color=:forestgreen, linewidth=4.0, label=p0 != 0.0f0 ? L"g_3(J) - p_0(J-1)" : L"g_3(J)")
            ref_vol = 5.0f0 .* (xs3_net .- 1.0f0) .^ 2 .+ h3_0
            #lines!(ax3, xs3_net, ref_vol, color=:crimson, linestyle=:dot, linewidth=5, label=L"D(J-1)^2 + g_3(1)")
            axislegend(ax3, position=:lt, framevisible=false)
        else
            lines!(ax3, xs3_net, h3_out, color=:forestgreen, linewidth=4.0, label=L"g_3(J)")
            axislegend(ax3, position=:lt, framevisible=false)
        end

        #Label(f_nets[4, mode],
        #    "S = $(round(S_mode, digits=1))  |  Share: $share_pct%\nRegime: $regime_str",
        #    fontsize=20, font=:bold)
    end

    rowsize!(f_nets.layout, 1, Auto())
    for col in 1:total_cols
        colsize!(f_nets.layout, col, Relative(1 / total_cols))
    end

    return f_nets
end

###############################################################################################################
# 2. 2D Strain Energy Landscapes Ψ(I1, I2) 
###############################################################################################################
function plot_2d_energy_landscape(models, raw_scale, num_modes; n_pts=100)
    xs1, xs2, _ = get_domain_ranges(n_pts)
    x1_mat = [i1 for i1 in xs1, i2 in xs2]
    x2_mat = [i2 for i1 in xs1, i2 in xs2]

    j_slices = [0.5f0, 1.0f0, 2.0f0]
    col_titles = ["Reference Ψ_true", "NN Prediction Ψ_NN", "Rel. Error |ΔΨ| / Ψ_true"]

    fig_2d = Figure(size=(1550, 1180), fontsize=15, figure_padding=(25, 25, 25, 55))

    # 1. Top Column Headers (Table Style)
    for (c_idx, title_str) in enumerate(col_titles)
        Label(fig_2d[1, c_idx+1], title_str, font=:bold, fontsize=18, tellwidth=false)
    end
    Label(fig_2d[1, 5], "Ψ", font=:bold, fontsize=18, tellwidth=false)
    Label(fig_2d[1, 6], "Rel. Err", font=:bold, fontsize=18, tellwidth=false)

    for (iter, j_slice) in enumerate(j_slices)
        row_num = iter + 1

        # Row Header on the left
        Label(fig_2d[row_num, 1], "J = $j_slice", font=:bold, fontsize=18, tellheight=false)

        x3_mat = fill(Float32(j_slice), size(x1_mat))
        x_flat = [vec(x1_mat)'; vec(x2_mat)'; vec(x3_mat)']

        # True reference energy
        psi_true = reshape(vec(reference_function.(eachcol(x_flat))), length(xs1), length(xs2))

        # Total NN energy
        psi_nn = reshape(vec(predict_all(models, raw_scale, x_flat)), length(xs1), length(xs2))

        # Mask non-admissible domain points with NaN
        for idx in eachindex(x1_mat)
            if !is_admissible(x1_mat[idx], x2_mat[idx], Float32(j_slice))
                psi_true[idx] = NaN32
                psi_nn[idx] = NaN32
            end
        end

        valid_true = filter(!isnan, psi_true)
        valid_nn = filter(!isnan, psi_nn)

        # Relative error with small epsilon to prevent 0/0 division at reference state (3, 3, 1)
        psi_err = abs.(psi_true .- psi_nn) ./ max.(abs.(psi_true), 1.0f-4)
        for idx in eachindex(x1_mat)
            if !is_admissible(x1_mat[idx], x2_mat[idx], Float32(j_slice))
                psi_err[idx] = NaN32
            end
        end

        valid_err = filter(!isnan, psi_err)

        cmin = isempty(valid_true) || isempty(valid_nn) ? 0f0 : min(minimum(valid_true), minimum(valid_nn))
        cmax = isempty(valid_true) || isempty(valid_nn) ? 1f0 : max(maximum(valid_true), maximum(valid_nn))
        err_max = isempty(valid_err) ? 1.0f0 : max(0.1f0, min(0.2f0, maximum(valid_err)))

        # 1. True Function (Axis preserving scaling)
        ax_true = Axis(fig_2d[row_num, 2], xlabel="I₁", ylabel="I₂", aspect=DataAspect())
        hm_true = heatmap!(ax_true, xs1, xs2, psi_true, colormap=:viridis, colorrange=(cmin, cmax), nan_color=:transparent)

        # 2. Total NN Prediction (shares colorrange with Reference)
        ax_nn = Axis(fig_2d[row_num, 3], xlabel="I₁", aspect=DataAspect())
        hm_nn = heatmap!(ax_nn, xs1, xs2, psi_nn, colormap=:viridis, colorrange=(cmin, cmax), nan_color=:transparent)

        # 3. Relative Error
        ax_err = Axis(fig_2d[row_num, 4], xlabel="I₁", aspect=DataAspect())
        hm_err = heatmap!(ax_err, xs1, xs2, psi_err, colormap=:inferno, colorrange=(0f0, err_max), nan_color=:transparent)

        # Indicate boundary of computed training data (I1 <= 20, I2 <= 20) via two dotted lines
        for ax in (ax_true, ax_nn, ax_err)
            vlines!(ax, [20.0f0], color=:black, linestyle=:dot, linewidth=2.0)
            hlines!(ax, [20.0f0], color=:black, linestyle=:dot, linewidth=2.0)
            xlims!(ax, Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]))
            ylims!(ax, Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]))
        end

        # Colorbars on the right of the figure for each row
        Colorbar(fig_2d[row_num, 5], hm_true, width=14, ticklabelsize=12)
        Colorbar(fig_2d[row_num, 6], hm_err, width=14, ticklabelsize=12)
    end

    # Precise column sizing to guarantee uniform plot dimensions
    colsize!(fig_2d.layout, 1, Fixed(85))
    for c in 2:4
        colsize!(fig_2d.layout, c, Fixed(345))
    end
    colsize!(fig_2d.layout, 5, Fixed(45))
    colsize!(fig_2d.layout, 6, Fixed(55))

    # Row sizing
    rowsize!(fig_2d.layout, 1, Fixed(45))
    for r in 2:(length(j_slices)+1)
        rowsize!(fig_2d.layout, r, Fixed(220))
    end

    # Layout spacing
    colgap!(fig_2d.layout, 1, 15)
    colgap!(fig_2d.layout, 2, 20)
    colgap!(fig_2d.layout, 3, 20)
    colgap!(fig_2d.layout, 4, 30)
    colgap!(fig_2d.layout, 5, 20)

    rowgap!(fig_2d.layout, 1, 25)
    for g in 2:length(j_slices)
        rowgap!(fig_2d.layout, g, 15)
    end

    return fig_2d
end

"""
Compute the maximum principal stretch λ₁ for a given (I₁, I₂, J).
λ₁ is the square root of the largest positive real root of:
ξ³ - I₁ ξ² + I₂ ξ - J² = 0.
"""
function principal_stretch_max(I1, I2, J)
    p = I2 - I1^2 / 3.0f0
    q = -J^2 + I1 * I2 / 3.0f0 - 2.0f0 * I1^3 / 27.0f0
    val = -p / 3.0f0
    if val <= 0f0
        return sqrt(max(1f-6, I1 / 3.0f0))
    end
    r = 2.0f0 * sqrt(val)
    arg = clamp(3.0f0 * q / (2.0f0 * p * sqrt(val)), -1.0f0, 1.0f0)
    theta = acos(arg) / 3.0f0

    xi1 = I1 / 3.0f0 + r * cos(theta)
    xi2 = I1 / 3.0f0 + r * cos(theta - 2.0f0 * Float32(pi) / 3.0f0)
    xi3 = I1 / 3.0f0 + r * cos(theta + 2.0f0 * Float32(pi) / 3.0f0)

    return sqrt(max(1f-6, max(xi1, xi2, xi3)))
end


###############################################################################################################
# 2b. 2D Nominal Stress Landscapes P₁₁(I₁, I₂) 
###############################################################################################################
function plot_2d_stress_landscape(models, raw_scale, num_modes; n_pts=100)
    xs1, xs2, _ = get_domain_ranges(n_pts)
    x1_mat = [i1 for i1 in xs1, i2 in xs2]
    x2_mat = [i2 for i1 in xs1, i2 in xs2]

    j_slices = [0.5f0, 1.0f0, 2.0f0]
    col_titles = ["Reference P₁₁", "NN Prediction P₁₁", "Rel. Error |ΔP₁₁| / P₁₁"]

    fig_stress = Figure(size=(1550, 1180), fontsize=15, figure_padding=(25, 25, 25, 55))

    # Top Column Headers
    for (c_idx, title_str) in enumerate(col_titles)
        Label(fig_stress[1, c_idx+1], title_str, font=:bold, fontsize=18, tellwidth=false)
    end
    Label(fig_stress[1, 5], "P₁₁", font=:bold, fontsize=18, tellwidth=false)
    Label(fig_stress[1, 6], "Rel. Err", font=:bold, fontsize=18, tellwidth=false)

    # Reference hydrostatic pressure at undeformed reference configuration X_REF = [3.0; 3.0; 1.0;;]:
    # p₀ = 2(∂Ψ/∂I₁) + 4(∂Ψ/∂I₂) + ∂Ψ/∂J
    # Subtracting p₀ from ∂Ψ/∂J ensures a stress-free reference configuration.
    grad_ref_0 = grad_psi_ref(X_REF[:, 1])
    p0_ref = 2f0 * grad_ref_0[1] + 4f0 * grad_ref_0[2] + grad_ref_0[3]

    p0_nn = p0

    for (iter, j_slice) in enumerate(j_slices)
        row_num = iter + 1

        Label(fig_stress[row_num, 1], "J = $j_slice", font=:bold, fontsize=18, tellheight=false)

        x3_mat = fill(Float32(j_slice), size(x1_mat))
        x_flat = [vec(x1_mat)'; vec(x2_mat)'; vec(x3_mat)']

        # Principal stretches λ₁
        lam1_flat = [principal_stretch_max(x_flat[1, i], x_flat[2, i], x_flat[3, i]) for i in 1:size(x_flat, 2)]

        # Reference gradients
        grad_ref = [grad_psi_ref(x_flat[:, i]) for i in 1:size(x_flat, 2)]
        dI1_ref = Float32[g[1] for g in grad_ref]
        dI2_ref = Float32[g[2] for g in grad_ref]
        dJ_ref = Float32[g[3] for g in grad_ref]

        # NN gradients
        dI1_nn, dI2_nn, dJ_nn = grad_psi_nn_batch(models, raw_scale, x_flat)

        I1_flat = vec(x1_mat)
        J_val = Float32(j_slice)

        # First Piola-Kirchhoff stress (corrected for stress-free reference state):
        # P₁₁ = 2*λ₁*(∂Ψ/∂I1) + 2*λ₁*(I1 - λ₁²)*(∂Ψ/∂I2) + (J/λ₁)*(∂Ψ/∂J - p₀)
        P11_true_flat = 2f0 .* lam1_flat .* dI1_ref .+ 2f0 .* lam1_flat .* (I1_flat .- lam1_flat .^ 2) .* dI2_ref .+ (J_val ./ lam1_flat) .* (dJ_ref .- p0_ref)
        P11_nn_flat = 2f0 .* lam1_flat .* dI1_nn .+ 2f0 .* lam1_flat .* (I1_flat .- lam1_flat .^ 2) .* dI2_nn .+ (J_val ./ lam1_flat) .* (dJ_nn .- p0_nn)

        p11_true = reshape(P11_true_flat, length(xs1), length(xs2))
        p11_nn = reshape(P11_nn_flat, length(xs1), length(xs2))

        # Mask non-admissible points
        for idx in eachindex(x1_mat)
            if !is_admissible(x1_mat[idx], x2_mat[idx], J_val)
                p11_true[idx] = NaN32
                p11_nn[idx] = NaN32
            end
        end

        valid_true = filter(!isnan, p11_true)
        valid_nn = filter(!isnan, p11_nn)

        # Regularized relative error: avoid division by zero near zero-stress reference state
        # eps_reg = 0.05 * max(|P11_true|) ensures smooth scaling without singularity
        max_p11 = isempty(valid_true) ? 1.0f0 : maximum(abs.(valid_true))
        eps_reg = 0.05f0 * max_p11
        p11_err = abs.(p11_true .- p11_nn) ./ max.(abs.(p11_true), eps_reg)
        for idx in eachindex(x1_mat)
            if !is_admissible(x1_mat[idx], x2_mat[idx], J_val)
                p11_err[idx] = NaN32
            end
        end

        valid_err = filter(!isnan, p11_err)

        cmin = isempty(valid_true) || isempty(valid_nn) ? 0f0 : min(minimum(valid_true), minimum(valid_nn))
        cmax = isempty(valid_true) || isempty(valid_nn) ? 1f0 : max(maximum(valid_true), maximum(valid_nn))
        err_max = isempty(valid_err) ? 1.0f0 : max(0.5f0, min(2.0f0, maximum(valid_err)))

        # 1. Reference P₁₁
        ax_true = Axis(fig_stress[row_num, 2], xlabel="I₁", ylabel="I₂", aspect=DataAspect())
        hm_true = heatmap!(ax_true, xs1, xs2, p11_true, colormap=:viridis, colorrange=(cmin, cmax), nan_color=:transparent)

        # 2. NN Prediction P₁₁
        ax_nn = Axis(fig_stress[row_num, 3], xlabel="I₁", aspect=DataAspect())
        hm_nn = heatmap!(ax_nn, xs1, xs2, p11_nn, colormap=:viridis, colorrange=(cmin, cmax), nan_color=:transparent)

        # 3. Relative Error
        ax_err = Axis(fig_stress[row_num, 4], xlabel="I₁", aspect=DataAspect())
        hm_err = heatmap!(ax_err, xs1, xs2, p11_err, colormap=:inferno, colorrange=(0f0, err_max), nan_color=:transparent)

        # Dotted boundary lines at (20, 20)
        for ax in (ax_true, ax_nn, ax_err)
            vlines!(ax, [20.0f0], color=:black, linestyle=:dot, linewidth=2.0)
            hlines!(ax, [20.0f0], color=:black, linestyle=:dot, linewidth=2.0)
            xlims!(ax, Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]))
            ylims!(ax, Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]))
        end

        # Colorbars
        Colorbar(fig_stress[row_num, 5], hm_true, width=14, ticklabelsize=12)
        Colorbar(fig_stress[row_num, 6], hm_err, width=14, ticklabelsize=12)
    end

    # Precise column sizing
    colsize!(fig_stress.layout, 1, Fixed(85))
    for c in 2:4
        colsize!(fig_stress.layout, c, Fixed(345))
    end
    colsize!(fig_stress.layout, 5, Fixed(45))
    colsize!(fig_stress.layout, 6, Fixed(55))

    # Row sizing
    rowsize!(fig_stress.layout, 1, Fixed(45))
    for r in 2:(length(j_slices)+1)
        rowsize!(fig_stress.layout, r, Fixed(220))
    end

    # Layout spacing
    colgap!(fig_stress.layout, 1, 15)
    colgap!(fig_stress.layout, 2, 20)
    colgap!(fig_stress.layout, 3, 20)
    colgap!(fig_stress.layout, 4, 30)
    colgap!(fig_stress.layout, 5, 20)

    rowgap!(fig_stress.layout, 1, 25)
    for g in 2:length(j_slices)
        rowgap!(fig_stress.layout, g, 15)
    end

    return fig_stress
end

###############################################################################################################
# 3. Dedicated Convergence History Plot (Train & Test Residuals)
###############################################################################################################
function plot_convergence(res_train, res_test; n_epochs_per_iter=500)
    fig = Figure(size=(850, 600), fontsize=26)
    ax = Axis(fig[1, 1],
        title="Training & Test Convergence History",
        xlabel=L"\textrm{Iteration }[\times 10^3]",
        ylabel="Residual",
        yscale=log10,
        xgridvisible=true,
        ygridvisible=true)

    iters_k = (1:length(res_train)) ./ 1000.0f0

    lines!(ax, iters_k, res_train, color=:royalblue, linewidth=3.5, label="Train Residual")
    #= if res_test !== nothing && !isempty(res_test)
        test_iters_k = (1:length(res_test)) ./ 1000.0f0
        lines!(ax, test_iters_k, res_test, color=:crimson, linestyle=:dash, linewidth=3.5, label="Test Residual")
    end =#

    #axislegend(ax, position=:rt, framevisible=true, backgroundcolor=(:white, 0.85))

    return fig
end

# Generate and display figures
println("\nGenerating 1D Sub-network visualization...")
fig_1d = plot_1d_subnetworks(models, raw_scale, nets_x1, nets_x2, nets_x3, num_modes)
display(fig_1d)

println("Generating 2D Energy landscape visualization (slice at J=1.0)...")
fig_2d = plot_2d_energy_landscape(models, raw_scale, num_modes)
display(fig_2d)

println("Generating 2D Stress landscape visualization (P₁₁ across J slices)...")
fig_stress = plot_2d_stress_landscape(models, raw_scale, num_modes)
display(fig_stress)

# Save figures to plots/Visualizations/<folder_name>
save_dir_plots = plotsdir("Visualizations", folder_name)
mkpath(save_dir_plots)

path_1d = joinpath(save_dir_plots, "Neo_Hook_Model_Architecture_1D_Subnetworks.png")
save(path_1d, fig_1d)
printstyled("Saved 1D sub-networks figure to: $path_1d\n", color=:green, bold=true)

path_2d = joinpath(save_dir_plots, "Neo_Hook_Model_Architecture_2D_Energy_Landscape.png")
save(path_2d, fig_2d)
printstyled("Saved 2D energy landscape figure to: $path_2d\n", color=:green, bold=true)

path_stress = joinpath(save_dir_plots, "Neo_Hook_Model_Architecture_2D_Stress_Landscape.png")
save(path_stress, fig_stress)
printstyled("Saved 2D stress landscape figure to: $path_stress\n", color=:green, bold=true)

if residual_over_iteration !== nothing && length(residual_over_iteration) > 0
    println("Generating Convergence History figure (Train & Test residuals)...")
    fig_conv = plot_convergence(residual_over_iteration, test_residual_over_iteration)
    display(fig_conv)
    path_conv = joinpath(save_dir_plots, "Neo_Hook_Convergence_History.png")
    save(path_conv, fig_conv)
    printstyled("Saved Convergence History figure to: $path_conv\n", color=:green, bold=true)
end