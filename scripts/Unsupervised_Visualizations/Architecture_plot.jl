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
# 1. Reference Strain Energy Function
###############################################################################################################
using JLD2

# Domain intervals for x1, x2, and x3 (matching FE experimental kinematics)
const X1_BOUNDS = (3.0, 12.0)
const X2_BOUNDS = (3.0, 10.0)
const X3_BOUNDS = (0.5, 1.8)
const X_REF = Float32[3.0; 3.0; 1.0;;]

# Load true simulation parameters if available, else default to compressible Neo-Hookean
const data_sim_path = datadir("sims", "euclid_identification_data.jld2")
if isfile(data_sim_path)
    data_sim = wload(data_sim_path)
    if haskey(data_sim, "material_params")
        const μ_true = Float32(data_sim["material_params"]["μ"])
        const λ_true = Float32(data_sim["material_params"]["λ"])
        reference_function(x) = (μ_true / 2.0f0) * (x[1] - 3.0f0 - 2.0f0 * log(max(1f-6, x[3]))) + (λ_true / 2.0f0) * (x[3] - 1.0f0)^2
    else
        reference_function(x) = 0.5f0 * (x[1] * x[3]^(-2 / 3) - 3f0) + 0.2f0 * (x[2] * x[3]^(-4 / 3) - 3f0) + 5.0f0 * (x[3] - 1)^2
    end
else
    reference_function(x) = 0.5f0 * (x[1] * x[3]^(-2 / 3) - 3f0) + 0.2f0 * (x[2] * x[3]^(-4 / 3) - 3f0) + 5.0f0 * (x[3] - 1)^2
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

get_domain_ranges(num_points_per_dim) = (
    range(Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]), length=num_points_per_dim),
    range(Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]), length=num_points_per_dim),
    range(Float32(X3_BOUNDS[1]), Float32(X3_BOUNDS[2]), length=num_points_per_dim),
)

split_input(x) = (x[1:1, :], x[2:2, :], x[3:3, :])
get_scale(scale_param, m=1) = length(scale_param) == 1 ? (softplus(scale_param[1]) + 1.0f0) : (softplus(scale_param[m]) + 1.0f0)

"""
Load trained Neo-Hookean model from EUCLID SparseDecomp or Final directory
"""
function load_network()
    # Check specific candidates first
    cand_dirs = [
        datadir("exp_raw", "EUCLID_SparseDecomp", "network.jld2"),
        datadir("exp_raw", "Final", "EUCLID_SparseDecomp_Noise", "network.jld2"),
    ]
    for c in cand_dirs
        if isfile(c)
            println("Loading network from: $c")
            return wload(c), basename(dirname(c))
        end
    end

    exp_raw_dir = datadir("exp_raw")
    subdirs = filter(d -> isdir(joinpath(exp_raw_dir, d)) && isfile(joinpath(exp_raw_dir, d, "network.jld2")), readdir(exp_raw_dir))
    if isempty(subdirs)
        error("No trained network found in $(exp_raw_dir). Please run training first.")
    end
    # Prioritize EUCLID_SparseDecomp runs, then D3 / NeoHook runs
    euclid_dirs = filter(d -> occursin("EUCLID_SparseDecomp", d), subdirs)
    d3_dirs = filter(d -> occursin("D3_NeoHook", d) || occursin("NeoHook", d) || occursin("C3_NeoHook", d), subdirs)
    target_dirs = !isempty(euclid_dirs) ? euclid_dirs : (!isempty(d3_dirs) ? d3_dirs : subdirs)
    sort!(target_dirs)
    latest_dir = joinpath(exp_raw_dir, target_dirs[end])
    println("Loading trained model from: $latest_dir")
    return wload(joinpath(latest_dir, "network.jld2")), target_dirs[end]
end

model_data, run_name = load_network()
raw_scale = model_data["raw_scale"]
num_modes = model_data["num_modes"]
const p0 = Float32(get(model_data, "p0", 0.0f0))
const is_d3_formulation = get(model_data, "formulation", "") == "d3" || haskey(model_data, "step_indices") || haskey(model_data, "p0") || haskey(model_data, "loss_history")

function repair_subnet(chain)
    new_layers = []
    for l in chain.layers
        if l isa Dense
            push!(new_layers, l)
        elseif hasproperty(l, :weight) && hasproperty(l, :bias)
            W = Matrix{Float32}(l.weight)
            if l.bias isa AbstractVector
                b = Vector{Float32}(l.bias)
                is_sp = hasproperty(l, :σ) && occursin("softplus", string(l.σ))
                act = is_sp ? Flux.softplus : identity
                push!(new_layers, Dense(W, b, act))
            else
                push!(new_layers, Dense(W, false, identity))
            end
        else
            push!(new_layers, l)
        end
    end
    return Chain(new_layers...)
end

if haskey(model_data, "nets_x1")
    nets_x1 = [repair_subnet(model_data["nets_x1"][i]) for i in 1:num_modes]
    nets_x2 = [repair_subnet(model_data["nets_x2"][i]) for i in 1:num_modes]
    nets_x3 = [repair_subnet(model_data["nets_x3"][i]) for i in 1:num_modes]
    models = [Chain(split_input, Parallel(+, nets_x1[i], nets_x2[i], nets_x3[i])) for i in 1:num_modes]
elseif haskey(model_data, "models")
    raw_models = model_data["models"]
    get_parallel(m) = hasproperty(m.layers[2], :layers) ? m.layers[2].layers : m.layers[2]
    nets_x1 = [repair_subnet(get_parallel(raw_models[i])[1]) for i in 1:num_modes]
    nets_x2 = [repair_subnet(get_parallel(raw_models[i])[2]) for i in 1:num_modes]
    nets_x3 = [repair_subnet(get_parallel(raw_models[i])[3]) for i in 1:num_modes]
    models = [Chain(split_input, Parallel(+, nets_x1[i], nets_x2[i], nets_x3[i])) for i in 1:num_modes]
else
    error("Network structure not recognized: neither 'nets_x1' nor 'models' found in checkpoint.")
end

function predict_all(models, raw_scale, x)
    psi_raw = sum(
        let S = get_scale(raw_scale, m)
            if is_d3_formulation
                S .* (exp.(models[m](x) ./ S) .- exp.(models[m](X_REF) ./ S))
            else
                S .* (exp.((models[m](x) .- models[m](X_REF)) ./ S) .- 1.0f0)
            end
        end for m in 1:length(models)
    )
    J = x[3:3, :]
    return psi_raw .- p0 .* (J .- 1.0f0)
end

residual_over_iteration = get(model_data, "residual_over_iteration", get(model_data, "loss_history", nothing))
test_residual_over_iteration = get(model_data, "test_residual_over_iteration", nothing)

println("Loaded model with $num_modes modes from $run_name.")

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
            if is_d3_formulation
                vec(S .* (exp.(models[k](x_3d) ./ S) .- exp.(models[k](X_REF) ./ S)))
            else
                vec(S .* (exp.((models[k](x_3d) .- models[k](X_REF)) ./ S) .- 1.0f0))
            end
        end for k in 1:num_modes
    ]
    mode_norms_global = [norm(mp, 2) for mp in mode_preds_global]
    sum_norms_global = sum(mode_norms_global)

    active_modes = num_modes
    total_cols = active_modes

    f_nets = Figure(size=(max(900, 480 * total_cols), 1350))
    #Label(f_nets[0, :], "Sub-network outputs per mode", fontsize=24, font=:bold)

    ref_x1 = Float32[3.0f0;;]
    ref_x2 = Float32[3.0f0;;]
    ref_x3 = Float32[1.0f0;;]

    for mode in 1:active_modes
        S_mode = get_scale(raw_scale, mode)

        ref1_val = nets_x1[mode](ref_x1)[1]
        ref2_val = nets_x2[mode](ref_x2)[1]
        ref3_val = nets_x3[mode](ref_x3)[1]

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
        if is_act1
            ref_c10 = (μ_true / 2.0f0) .* (xs1_net .- 3.0f0) .+ h1_0
            lines!(ax1, xs1_net, ref_c10, color=:crimson, linestyle=:dot, linewidth=5, label=L"\mu/2(I_1-3) + g_1(3)")
        else
            ylims!(ax1, 0.0, 2.0)
        end
        xlims!(ax1, Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]))
        axislegend(ax1, position=:lt, framevisible=false)

        # 2. Net x2 (I2)
        ax2 = Axis(f_nets[3, mode],
            title="Net I₂",
            xlabel="I₂",
            xgridvisible=true,
            ygridvisible=true)
        lines!(ax2, xs2_net, h2_out, color=:orange, linewidth=4.0, label=L"g_2(I_2)")
        if !is_act2
            ylims!(ax2, 0.0, 2.0)
        end
        xlims!(ax2, Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]))
        axislegend(ax2, position=:lt, framevisible=false)

        # 3. Net x3 (J)
        ax3 = Axis(f_nets[4, mode],
            title="Net J",
            xlabel="J",
            xgridvisible=true,
            ygridvisible=true)
        if is_act3
            h3_phys = h3_out .- p0 .* (xs3_net .- 1.0f0)
            lines!(ax3, xs3_net, h3_phys, color=:forestgreen, linewidth=4.0, label=p0 != 0.0f0 ? L"g_3(J) - p_0(J-1)" : L"g_3(J)")
            if !is_act1 && !is_act2
                ref_vol_true = @. -μ_true * log(max(1f-6, xs3_net)) + (λ_true / 2.0f0) * (xs3_net - 1.0f0)^2 + h3_0
                lines!(ax3, xs3_net, ref_vol_true, color=:crimson, linestyle=:dot, linewidth=5)
                axislegend(ax3,
                    [
                        LineElement(color=:forestgreen, linewidth=4.0),
                        LineElement(color=:crimson, linestyle=:dot, linewidth=5),
                        LineElement(color=:transparent)
                    ],
                    [
                        p0 != 0.0f0 ? L"g_3(J) - p_0(J-1)" : L"g_3(J)",
                        L"-\mu \ln(J)",
                        L"+ \lambda/2(J-1)^2 + g_3(1)"
                    ],
                    position=:rt, framevisible=false, rowgap=2)
            elseif is_act1 && !is_act2
                j_pow = xs3_net .^ (-2 / 3f0)
                A_fit = [j_pow ones(Float32, length(j_pow))]
                ab_fit = A_fit \ h3_out
                j_ref = ab_fit[1] .* j_pow .+ ab_fit[2]
                lines!(ax3, xs3_net, j_ref, color=:crimson, linestyle=:dot, linewidth=5, label=L"J^{-2/3}")
                axislegend(ax3, position=:rt, framevisible=false)
            elseif !is_act1 && is_act2
                j_pow = xs3_net .^ (-4 / 3f0)
                A_fit = [j_pow ones(Float32, length(j_pow))]
                ab_fit = A_fit \ h3_out
                j_ref = ab_fit[1] .* j_pow .+ ab_fit[2]
                lines!(ax3, xs3_net, j_ref, color=:crimson, linestyle=:dot, linewidth=5, label=L"J^{-4/3}")
                axislegend(ax3, position=:rt, framevisible=false)
            else
                axislegend(ax3, position=:lt, framevisible=false)
            end
        else
            lines!(ax3, xs3_net, h3_out, color=:forestgreen, linewidth=4.0, label=L"g_3(J)")
            ylims!(ax3, 0.0, 2.0)
            axislegend(ax3, position=:lt, framevisible=false)
        end
        xlims!(ax3, Float32(X3_BOUNDS[1]), Float32(X3_BOUNDS[2]))

        # Row 4: Mode Share and Scale Info
        #mode_share_global = mode_norms_global[mode] / max(sum_norms_global, 1f-8)
        #Label(f_nets[4, mode], "scale = $(round(S_mode, digits=2))\nMode Share = $(round(mode_share_global * 100, digits=2))%", fontsize=16, font=:bold)
    end

    rowsize!(f_nets.layout, 1, Auto())
    for col in 1:total_cols
        colsize!(f_nets.layout, col, Relative(1 / total_cols))
    end

    return f_nets
end

###############################################################################################################
# 1b. Overlay Plot of Individual Sub-Networks across All Modes
###############################################################################################################
function plot_individual_nets_overlay(nets_x1, nets_x2, nets_x3, num_modes, raw_scale; n_pts=200)
    xs1, xs2, xs3 = get_domain_ranges(n_pts)
    x1_mat = collect(xs1)'
    x2_mat = collect(xs2)'
    x3_mat = collect(xs3)'

    fig = Figure(size=(1650, 520), fontsize=20)

    ax1 = Axis(fig[1, 1], title="Individual Nets: I₁ across Modes", xlabel="I₁", ylabel="Subnet Contribution Δf",
        xgridvisible=true, ygridvisible=true)
    ax2 = Axis(fig[1, 2], title="Individual Nets: I₂ across Modes", xlabel="I₂", ylabel="Subnet Contribution Δf",
        xgridvisible=true, ygridvisible=true)
    ax3 = Axis(fig[1, 3], title="Individual Nets: J across Modes", xlabel="J", ylabel="Subnet Contribution Δf",
        xgridvisible=true, ygridvisible=true)

    colors = [:dodgerblue, :crimson, :darkorange, :purple, :forestgreen, :teal, :magenta]

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

        #if w3_1 > 0.001f0
        lines!(ax1, xs1, df1, color=c, linewidth=3.5, label="Mode $m (W₃=$(round(w3_1, digits=2)))")
        #end
        #if w3_2 > 0.001f0
        lines!(ax2, xs2, df2, color=c, linewidth=3.5, label="Mode $m (W₃=$(round(w3_2, digits=2)))")
        #end
        #if w3_3 > 0.001f0
        lines!(ax3, xs3, df3, color=c, linewidth=3.5, label="Mode $m (W₃=$(round(w3_3, digits=2)))")
        #end
    end

    axislegend(ax1, position=:lt, framevisible=true)
    axislegend(ax2, position=:lt, framevisible=true)
    axislegend(ax3, position=:lt, framevisible=true)

    return fig
end

###############################################################################################################
# 2. 2D Strain Energy Landscapes Ψ(I1, I2) 
###############################################################################################################
function plot_2d_energy_landscape(models, raw_scale, num_modes; n_pts=60)
    xs1, xs2, _ = get_domain_ranges(n_pts)
    x1_mat = [i1 for i1 in xs1, i2 in xs2]
    x2_mat = [i2 for i1 in xs1, i2 in xs2]

    j_slices = [0.95f0, 1.0f0, 1.15f0, 1.30f0]
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

        # Mask non-admissible domain points with NaN (only plot physically admissible region)
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
        err_max = isempty(valid_err) ? 1.0f0 : max(0.1f0, min(1.0f0, maximum(valid_err)))

        # 1. True Function (Axis preserving scaling)
        ax_true = Axis(fig_2d[row_num, 2], xlabel="I₁", ylabel="I₂", aspect=DataAspect())
        hm_true = heatmap!(ax_true, xs1, xs2, psi_true, colormap=:viridis, colorrange=(cmin, cmax), nan_color=:transparent)

        # 2. Total NN Prediction (shares colorrange with Reference)
        ax_nn = Axis(fig_2d[row_num, 3], xlabel="I₁", aspect=DataAspect())
        hm_nn = heatmap!(ax_nn, xs1, xs2, psi_nn, colormap=:viridis, colorrange=(cmin, cmax), nan_color=:transparent)

        # 3. Absolute Error
        ax_err = Axis(fig_2d[row_num, 4], xlabel="I₁", aspect=DataAspect())
        hm_err = heatmap!(ax_err, xs1, xs2, psi_err, colormap=:inferno, colorrange=(0f0, err_max), nan_color=:transparent)

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

"""
Analytical gradient of reference energy with respect to x = [I1, I2, J]:
[∂Ψ/∂I1, ∂Ψ/∂I2, ∂Ψ/∂J]
"""
function grad_psi_ref(x)
    I1, I2, J = x[1], x[2], x[3]
    if @isdefined(μ_true) && @isdefined(λ_true)
        d_I1 = μ_true / 2.0f0
        d_I2 = 0.0f0
        d_J = -μ_true / max(1f-6, J) + λ_true * (J - 1.0f0)
    else
        C10 = 0.5f0
        C01 = 0.2f0
        D = 5.0f0
        d_I1 = C10 * J^(-2 / 3f0)
        d_I2 = C01 * J^(-4 / 3f0)
        d_J = -(2 / 3f0) * C10 * I1 * J^(-5 / 3f0) - (4 / 3f0) * C01 * I2 * J^(-7 / 3f0) + 2f0 * D * (J - 1f0)
    end
    return Float32[d_I1, d_I2, d_J]
end

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
Compute batch gradients [∂Ψ_NN/∂I1, ∂Ψ_NN/∂I2, ∂Ψ_NN/∂J] for x_mat (3 x N).
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

        if is_d3_formulation
            exp_factor = vec(exp.(models[m](x_mat) ./ S_m))
        else
            exp_factor = vec(exp.((models[m](x_mat) .- models[m](X_REF)) ./ S_m))
        end

        du1 = vec(df_dx_subnet(net1, I1_row))
        du2 = vec(df_dx_subnet(net2, I2_row))
        du3 = vec(df_dx_subnet(net3, J_row))

        dPsi_dI1 .+= exp_factor .* du1
        dPsi_dI2 .+= exp_factor .* du2
        dPsi_dJ .+= exp_factor .* du3
    end

    return dPsi_dI1, dPsi_dI2, dPsi_dJ
end

###############################################################################################################
# 2b. 2D Nominal Stress Landscapes P₁₁(I₁, I₂) 
###############################################################################################################
function plot_2d_stress_landscape(models, raw_scale, num_modes; n_pts=60)
    xs1, xs2, _ = get_domain_ranges(n_pts)
    x1_mat = [i1 for i1 in xs1, i2 in xs2]
    x2_mat = [i2 for i1 in xs1, i2 in xs2]

    j_slices = [0.95f0, 1.0f0, 1.15f0, 1.30f0]
    col_titles = ["Reference P₁₁", "NN Prediction P₁₁", "Rel. Error |ΔP₁₁| / P₁₁"]

    fig_stress = Figure(size=(1550, 1180), fontsize=15, figure_padding=(25, 25, 25, 55))

    for (c_idx, title_str) in enumerate(col_titles)
        Label(fig_stress[1, c_idx+1], title_str, font=:bold, fontsize=18, tellwidth=false)
    end
    Label(fig_stress[1, 5], "P₁₁", font=:bold, fontsize=18, tellwidth=false)
    Label(fig_stress[1, 6], "Rel. Err", font=:bold, fontsize=18, tellwidth=false)

    for (iter, j_slice) in enumerate(j_slices)
        row_num = iter + 1
        Label(fig_stress[row_num, 1], "J = $j_slice", font=:bold, fontsize=18, tellheight=false)

        x3_mat = fill(Float32(j_slice), size(x1_mat))
        x_flat = [vec(x1_mat)'; vec(x2_mat)'; vec(x3_mat)']

        # Maximum principal stretch λ₁
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

        # First Piola-Kirchhoff stress P₁₁ (corrected for stress-free reference state):
        # P₁₁ = 2*λ₁*(∂Ψ/∂I₁) + 2*λ₁*(I₁ - λ₁²)*(∂Ψ/∂I₂) + (J/λ₁)*(∂Ψ/∂J - p₀)
        P11_true_flat = 2f0 .* lam1_flat .* dI1_ref .+ 2f0 .* lam1_flat .* (I1_flat .- lam1_flat .^ 2) .* dI2_ref .+ (J_val ./ lam1_flat) .* dJ_ref
        P11_nn_flat = 2f0 .* lam1_flat .* dI1_nn .+ 2f0 .* lam1_flat .* (I1_flat .- lam1_flat .^ 2) .* dI2_nn .+ (J_val ./ lam1_flat) .* (dJ_nn .- p0)

        p11_true = reshape(P11_true_flat, length(xs1), length(xs2))
        p11_nn = reshape(P11_nn_flat, length(xs1), length(xs2))

        # Mask non-admissible points with NaN (only plot physically admissible region)
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

        ax_true = Axis(fig_stress[row_num, 2], xlabel="I₁", ylabel="I₂", aspect=DataAspect())
        hm_true = heatmap!(ax_true, xs1, xs2, p11_true, colormap=:viridis, colorrange=(cmin, cmax), nan_color=:transparent)

        ax_nn = Axis(fig_stress[row_num, 3], xlabel="I₁", aspect=DataAspect())
        hm_nn = heatmap!(ax_nn, xs1, xs2, p11_nn, colormap=:viridis, colorrange=(cmin, cmax), nan_color=:transparent)

        ax_err = Axis(fig_stress[row_num, 4], xlabel="I₁", aspect=DataAspect())
        hm_err = heatmap!(ax_err, xs1, xs2, p11_err, colormap=:inferno, colorrange=(0f0, err_max), nan_color=:transparent)

        Colorbar(fig_stress[row_num, 5], hm_true, width=14, ticklabelsize=12)
        Colorbar(fig_stress[row_num, 6], hm_err, width=14, ticklabelsize=12)
    end

    colsize!(fig_stress.layout, 1, Fixed(85))
    for c in 2:4
        colsize!(fig_stress.layout, c, Fixed(345))
    end
    colsize!(fig_stress.layout, 5, Fixed(45))
    colsize!(fig_stress.layout, 6, Fixed(55))

    rowsize!(fig_stress.layout, 1, Fixed(45))
    for r in 2:(length(j_slices)+1)
        rowsize!(fig_stress.layout, r, Fixed(220))
    end

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
function plot_convergence(res_train, res_test=nothing; switch_step=1000, step_size_after=50)
    fig = Figure(size=(850, 600), fontsize=26)
    ax = Axis(fig[1, 1],
        title="Convergence History",
        xlabel=L"\textrm{Iteration }[\times 10^3]",
        ylabel="Residual",
        yscale=log10,
        xgridvisible=true,
        ygridvisible=true)

    compute_iters_k(n) =
        if n <= switch_step
            collect(Float32, 1:n) ./ 1000.0f0
        else
            vcat(collect(Float32, 1:switch_step), Float32[switch_step + (i - switch_step) * step_size_after for i in (switch_step+1):n]) ./ 1000.0f0
        end

    iters_k = compute_iters_k(length(res_train))

    lines!(ax, iters_k, res_train, color=:royalblue, linewidth=3.5)
    if res_test !== nothing && !isempty(res_test)
        test_iters_k = compute_iters_k(length(res_test))
        lines!(ax, test_iters_k, res_test, color=:crimson, linestyle=:dash, linewidth=3.5, label="Test Residual")
        axislegend(ax, position=:rt, framevisible=true, backgroundcolor=(:white, 0.85))
    end

    return fig
end

# Generate and display figures
println("\nGenerating 1D Sub-network visualization...")
fig_1d = plot_1d_subnetworks(models, raw_scale, nets_x1, nets_x2, nets_x3, num_modes)
display(fig_1d)

println("Generating Individual Nets Overlay visualization...")
fig_overlay = plot_individual_nets_overlay(nets_x1, nets_x2, nets_x3, num_modes, raw_scale)
display(fig_overlay)

println("Generating 2D Energy landscape visualization...")
fig_2d = plot_2d_energy_landscape(models, raw_scale, num_modes)
display(fig_2d)

println("Generating 2D Stress P₁₁ landscape visualization...")
fig_stress = plot_2d_stress_landscape(models, raw_scale, num_modes)
display(fig_stress)

# Save figures to plots/Visualizations
save_dir_plots = plotsdir("EUCLID")
mkpath(save_dir_plots)

path_1d = joinpath(save_dir_plots, "Model_Architecture_1D_Subnetworks.png")
save(path_1d, fig_1d)
save(plotsdir("Visualizations", "Neo_Hook_Model_Architecture_1D_Subnetworks.png"), fig_1d)
final_exp_dir = datadir("exp_raw", "Final", "EUCLID_SparseDecomp")
if isdir(final_exp_dir)
    save(joinpath(final_exp_dir, "Neo_Hook_Model_Architecture_1D_Subnetworks.png"), fig_1d)
end
printstyled("Saved 1D sub-networks figure to: $path_1d\n", color=:green, bold=true)

path_overlay = joinpath(save_dir_plots, "Individual_Nets_Overlay.png")
save(path_overlay, fig_overlay)
printstyled("Saved Individual Nets overlay figure to: $path_overlay\n", color=:green, bold=true)

path_2d = joinpath(save_dir_plots, "Model_Architecture_2D_Energy_Landscape.png")
save(path_2d, fig_2d)
printstyled("Saved 2D energy landscape figure to: $path_2d\n", color=:green, bold=true)

path_stress = joinpath(save_dir_plots, "Model_Architecture_2D_Stress_P11.png")
save(path_stress, fig_stress)
printstyled("Saved 2D stress P₁₁ landscape figure to: $path_stress\n", color=:green, bold=true)

if residual_over_iteration !== nothing && length(residual_over_iteration) > 0
    println("Generating Convergence History figure...")
    fig_conv = plot_convergence(residual_over_iteration, test_residual_over_iteration)
    display(fig_conv)
    path_conv = joinpath(save_dir_plots, "Convergence_History.png")
    save(path_conv, fig_conv)
    printstyled("Saved Convergence History figure to: $path_conv\n", color=:green, bold=true)
end