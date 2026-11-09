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

# Analytical reference function (Mooney-Rivlin / Polynomial strain energy)
const C10 = 0.5f0
const C02 = 0.2f0
reference_function(x) = C10 * (x[1] - 3.0f0)^2 + C02 * (x[2] - 3.0f0)

"""
Discriminant condition for real positive principal stretches in incompressible elasticity:
det(F) = 1 <=> lambda^3 - I1*lambda^2 + I2*lambda - 1 = 0 has 3 positive real roots.
"""
function is_admissible(I1, I2)
    delta = -4f0 * I1^3 - 4f0 * I2^3 + I1^2 * I2^2 + 18f0 * I1 * I2 - 27f0
    return delta >= -1f-3 && I1 >= 3.0f0 && I2 >= 3.0f0
end

# Incompressible domain bounds for invariants I1 and I2
const X1_BOUNDS = (3.0, 10.0)
const X2_BOUNDS = (3.0, 18.0)

# Reference state in invariant space (undeformed reference configuration: I1 = 3, I2 = 3)
const X_REF = Float32[3.0; 3.0;;]
const REF_X1 = Float32[3.0;;]
const REF_X2 = Float32[3.0;;]

get_domain_ranges(num_points_per_dim) = (
    range(Float32(X1_BOUNDS[1]), Float32(X1_BOUNDS[2]), length=num_points_per_dim),
    range(Float32(X2_BOUNDS[1]), Float32(X2_BOUNDS[2]), length=num_points_per_dim),
)

split_input(x) = (x[1:1, :], x[2:2, :])
get_scale(scale_param, m=1) = length(scale_param) == 1 ? (softplus(scale_param[1]) + 1.0f0) : (softplus(scale_param[m]) + 1.0f0)

predict_all(models, raw_scale, x) = sum(
    let S = get_scale(raw_scale, m)
        S .* (exp.(models[m](x) ./ S) .- exp.(models[m](X_REF) ./ S))
    end for m in 1:length(models)
)

"""
Load trained model from Final directory or latest exp_raw run
"""
function load_network()
    cand_paths = [
        datadir("exp_raw", "Final", "E1_incompressible", "network.jld2"),
        datadir("exp_raw", "Final", "E1_incompressible_RegHigh", "network.jld2"),
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
            return wload(joinpath(latest_dir, "network.jld2")), basename(latest_dir)
        end
    end
    error("No trained polynomial network found in $(datadir("exp_raw")). Please run training first (scripts/Supervised/Polynomial.jl).")
end

model_data, folder_name = load_network()
nets_x1 = model_data["nets_x1"]
nets_x2 = model_data["nets_x2"]
raw_scale = model_data["raw_scale"]
num_modes = model_data["num_modes"]
residual_over_iteration = get(model_data, "residual_over_iteration", nothing)
test_residual_over_iteration = get(model_data, "test_residual_over_iteration", nothing)
x_train = get(model_data, "x_train", nothing)
y_train = get(model_data, "y_train", nothing)
y_train_clean = get(model_data, "y_train_clean", nothing)
train_info = get(model_data, "train_info", nothing)
noise_level = get(model_data, "noise_level", nothing)

models = [Chain(split_input, Parallel(+, nets_x1[i], nets_x2[i])) for i in 1:num_modes]

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

function generate_special_deformation_data(n_per_mode=30; random_jitter=false, return_info=false)
    pts = Vector{Float32}[]
    stretches = Float32[]
    modes = Symbol[]
    add_jitter(r) = random_jitter ? r .+ 0.005f0 .* randn(Float32, length(r)) : r

    stretches_ut = add_jitter(range(0.6f0, 3.0f0, length=n_per_mode))
    for lam in stretches_ut
        push!(pts, deformation_invariants(:UT, lam))
        push!(stretches, lam)
        push!(modes, :UT)
    end

    stretches_ps = add_jitter(range(0.6f0, 2.5f0, length=n_per_mode))
    for lam in stretches_ps
        push!(pts, deformation_invariants(:PS, lam))
        push!(stretches, lam)
        push!(modes, :PS)
    end

    stretches_eb = add_jitter(range(0.6f0, 2.0f0, length=n_per_mode))
    for lam in stretches_eb
        push!(pts, deformation_invariants(:EB, lam))
        push!(stretches, lam)
        push!(modes, :EB)
    end

    x_mat = reduce(hcat, pts)
    return return_info ? (x_mat, (stretches=stretches, modes=modes)) : x_mat
end

if x_train === nothing || train_info === nothing
    x_train, train_info = generate_special_deformation_data(50; random_jitter=false, return_info=true)
    y_train = Float32.(vec(reference_function.(eachcol(x_train))))'
    y_train_clean = y_train
end

println("Loaded model with $num_modes modes from $folder_name.")

###############################################################################################################
# 1. 1D Sub-Network Outputs per Mode across X1_BOUNDS and X2_BOUNDS & Convergence
###############################################################################################################
function plot_1d_subnetworks(models, raw_scale, nets_x1, nets_x2, num_modes; n_pts=200)
    xs1_net, xs2_net = get_domain_ranges(n_pts)
    x1_net = collect(xs1_net)'
    x2_net = collect(xs2_net)'

    # Evaluate 2D grid for mode share computation
    xs1_grid, xs2_grid = get_domain_ranges(40)
    x1_2d = repeat(xs1_grid, inner=length(xs2_grid))
    x2_2d = repeat(xs2_grid, outer=length(xs1_grid))
    x_2d = [x1_2d'; x2_2d']

    mode_preds_global = [
        let S = get_scale(raw_scale, k)
            vec(S .* (exp.(models[k](x_2d) ./ S) .- exp.(models[k](X_REF) ./ S)))
        end for k in 1:num_modes
    ]
    mode_norms_global = [norm(mp, 2) for mp in mode_preds_global]
    sum_norms_global = sum(mode_norms_global)

    active_modes = num_modes
    total_cols = active_modes

    f_nets = Figure(size=(max(900, 480 * total_cols), 950))

    for mode in 1:active_modes
        S_mode = get_scale(raw_scale, mode)

        # Raw outputs u(I) before scale transformation (relative to reference state)
        ref1_val = nets_x1[mode](REF_X1)[1]
        ref2_val = nets_x2[mode](REF_X2)[1]
        raw1_out = vec(nets_x1[mode](x1_net) .- ref1_val)
        raw2_out = vec(nets_x2[mode](x2_net) .- ref2_val)

        is_act1 = maximum(abs, raw1_out) > 1e-3
        is_act2 = maximum(abs, raw2_out) > 1e-3

        # Scaling factor distribution:
        # If only one mode/sub-network is active, S is only used on this mode.
        # If both are active, sqrt(S) is used on both.
        # If neither is active, factor 1.0 is used.
        if is_act1 && !is_act2
            scale1 = S_mode
            scale2 = 1.0f0
        elseif !is_act1 && is_act2
            scale1 = 1.0f0
            scale2 = S_mode
        elseif is_act1 && is_act2
            scale1 = sqrt(S_mode)
            scale2 = sqrt(S_mode)
        else
            scale1 = 1.0f0
            scale2 = 1.0f0
        end

        # Sub-network outputs without energy normalization:
        h1_out = vec(scale1 .* exp.(nets_x1[mode](x1_net) ./ S_mode))
        h2_out = vec(scale2 .* exp.(nets_x2[mode](x2_net) ./ S_mode))

        h1_0 = scale1 * exp(ref1_val / S_mode)
        h2_0 = scale2 * exp(ref2_val / S_mode)

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
        if is_act1 && mode == 1
            ref_c10 = C10 .* (xs1_net .- 3.0f0) .^ 2 .+ h1_0
            lines!(ax1, xs1_net, ref_c10, color=:crimson, linestyle=:dot, linewidth=5, label=L"C_{20}(I_1-3)^2 + g_2(3)")
        end
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
        if is_act2 && mode == 2
            ref_c02 = C02 .* (xs2_net .- 3.0f0) .+ h2_0
            lines!(ax2, xs2_net, ref_c02, color=:crimson, linestyle=:dot, linewidth=5, label=L"C_{01}(I_2-3) + g_2(3)")
        end
        if !is_act2
            ylims!(ax2, 0.0, 2.0)
        end
        axislegend(ax2, position=:lt, framevisible=false)

        #=         # Stiffening ratio rho = max(u) / S and regime classification
                u_max = max(maximum(raw1_out), maximum(raw2_out))
                rho = round(u_max / S_mode, digits=2)
                regime_str = rho < 0.25f0 ? "Quasi-linear (u/S = $rho)" : "Exponential (u/S = $rho)"
                mode_share_global = mode_norms_global[mode] / max(sum_norms_global, 1f-8)
                share_pct = round(mode_share_global * 100, digits=1)

                Label(f_nets[3, mode],
                    "S = $(round(S_mode, digits=1))  |  Share: $share_pct%\nRegime: $regime_str",
                    fontsize=20, font=:bold) =#
    end

    #rowsize!(f_nets.layout, 2, Fixed(70))
    rowsize!(f_nets.layout, 1, Auto())
    for col in 1:total_cols
        colsize!(f_nets.layout, col, Relative(1 / total_cols))
    end

    return f_nets
end

###############################################################################################################
# 2. 2D Strain Energy Landscapes Ψ(I1, I2) across Admissible Incompressible Domain
###############################################################################################################
function plot_2d_energy_landscape(models, raw_scale, num_modes; n_pts=120)
    xs1, xs2 = get_domain_ranges(n_pts)
    x1_mat = [i1 for i1 in xs1, i2 in xs2]
    x2_mat = [i2 for i1 in xs1, i2 in xs2]
    x_flat = [vec(x1_mat)'; vec(x2_mat)']

    # True reference energy
    psi_true = reshape(vec(reference_function.(eachcol(x_flat))), length(xs1), length(xs2))

    # Total NN energy
    psi_nn = reshape(vec(predict_all(models, raw_scale, x_flat)), length(xs1), length(xs2))
    # Relative error with small epsilon to prevent 0/0 division at reference state (3, 3)
    psi_err = abs.(psi_true .- psi_nn) ./ max.(abs.(psi_true), 1.0f-4)

    # Mask non-admissible domain points with NaN
    for idx in eachindex(x1_mat)
        if !is_admissible(x1_mat[idx], x2_mat[idx])
            psi_true[idx] = NaN32
            psi_nn[idx] = NaN32
            psi_err[idx] = NaN32
        end
    end

    valid_true = filter(!isnan, psi_true)
    valid_nn = filter(!isnan, psi_nn)
    valid_err = filter(!isnan, psi_err)
    cmin = min(minimum(valid_true), minimum(valid_nn))
    cmax = max(maximum(valid_true), maximum(valid_nn))
    err_max = isempty(valid_err) ? 0.3f0 : max(0.005f0, min(1.0f0, maximum(valid_err)))

    total_cols = 3
    fig_2d = Figure(size=(min(1800, 380 * total_cols), 450), fontsize=18)

    # Deformation data trajectories from training (UT, PS, EB)
    # 1. Uniaxial (UT): lam in [0.6, 3.0]
    lams_ut = range(1.0f0, 3.0f0, length=200)
    lp_ut = 1.0f0 ./ sqrt.(lams_ut)
    i1_ut = lams_ut .^ 2 .+ 2.0f0 .* lp_ut .^ 2
    i2_ut = lp_ut .^ 4 .+ 2.0f0 .* lams_ut .^ 2 .* lp_ut .^ 2
    m_ut = (i1_ut .<= Float32(X1_BOUNDS[2])) .& (i2_ut .<= Float32(X2_BOUNDS[2]))

    # 2. Pure Shear (PS): lam in [0.6, 2.5]
    lams_ps = range(1.0f0, 2.7f0, length=200)
    l3_ps = 1.0f0 ./ lams_ps
    i1_ps = lams_ps .^ 2 .+ 1.0f0 .+ l3_ps .^ 2
    i2_ps = lams_ps .^ 2 .+ (lams_ps .* l3_ps) .^ 2 .+ l3_ps .^ 2
    m_ps = (i1_ps .<= Float32(X1_BOUNDS[2])) .& (i2_ps .<= Float32(X2_BOUNDS[2]))

    # 3. Equi-biaxial (EB): lam in [0.6, 2.0]
    lams_eb = range(1.0f0, 2.0f0, length=200)
    l3_eb = 1.0f0 ./ (lams_eb .^ 2)
    i1_eb = 2.0f0 .* lams_eb .^ 2 .+ l3_eb .^ 2
    i2_eb = lams_eb .^ 4 .+ 2.0f0 .* lams_eb .^ 2 .* l3_eb .^ 2
    m_eb = (i1_eb .<= Float32(X1_BOUNDS[2])) .& (i2_eb .<= Float32(X2_BOUNDS[2]))

    # 1. True Function
    ax_true = Axis(fig_2d[1, 1], title="Reference Ψ_true", xlabel="I₁", ylabel="I₂")
    hm_true = heatmap!(ax_true, xs1, xs2, psi_true, colormap=:viridis, colorrange=(cmin, cmax), nan_color=:transparent)
    Colorbar(fig_2d[2, 1], hm_true, vertical=false)

    # 2. Total NN Prediction
    ax_nn = Axis(fig_2d[1, 2], title="NN Prediction Ψ_NN", xlabel="I₁", ylabel="I₂")
    hm_nn = heatmap!(ax_nn, xs1, xs2, psi_nn, colormap=:viridis, colorrange=(cmin, cmax), nan_color=:transparent)
    Colorbar(fig_2d[2, 2], hm_nn, vertical=false)

    # 3. Relative Error
    ax_err = Axis(fig_2d[1, 3], title="Rel. Error |ΔΨ| / Ψ_true", xlabel="I₁", ylabel="I₂")
    hm_err = heatmap!(ax_err, xs1, xs2, psi_err, colormap=:inferno, nan_color=:transparent, colorrange=(0.0, err_max))
    Colorbar(fig_2d[2, 3], hm_err, vertical=false)

    # Spacing for markers along trajectories
    step_ut = max(1, count(m_ut) ÷ 12)
    step_ps = max(1, count(m_ps) ÷ 10)
    step_eb = max(1, count(m_eb) ÷ 12)

    i1_ut_v, i2_ut_v = i1_ut[m_ut], i2_ut[m_ut]
    i1_ps_v, i2_ps_v = i1_ps[m_ps], i2_ps[m_ps]
    i1_eb_v, i2_eb_v = i1_eb[m_eb], i2_eb[m_eb]

    # Overlay training deformation mode trajectories on all subplots with spaced markers
    for ax in (ax_true, ax_nn, ax_err)
        lines!(ax, i1_ut_v, i2_ut_v, color=:crimson, linewidth=3.0)
        scatter!(ax, i1_ut_v[1:step_ut:end], i2_ut_v[1:step_ut:end], color=:crimson, marker=:circle, markersize=20)

        lines!(ax, i1_ps_v, i2_ps_v, color=:green, linewidth=3.0)
        scatter!(ax, i1_ps_v[1:step_ps:end], i2_ps_v[1:step_ps:end], color=:green, marker=:rect, markersize=20)

        lines!(ax, i1_eb_v, i2_eb_v, color=:royalblue, linewidth=3.0)
        scatter!(ax, i1_eb_v[1:step_eb:end], i2_eb_v[1:step_eb:end], color=:royalblue, marker=:utriangle, markersize=20)
    end

    legend_elems = [
        [LineElement(color=:crimson, linewidth=3.0), MarkerElement(marker=:circle, color=:crimson, markersize=20)],
        [LineElement(color=:green, linewidth=3.0), MarkerElement(marker=:rect, color=:green, markersize=20)],
        [LineElement(color=:royalblue, linewidth=3.0), MarkerElement(marker=:utriangle, color=:royalblue, markersize=20)],
    ]
    legend_labels = ["Uniaxial (UT)", "Pure Shear (PS)", "Equi-biaxial (EB)"]
    axislegend(ax_true, legend_elems, legend_labels, position=:lt, framevisible=true, backgroundcolor=(:white, 0.85), labelsize=14)

    for col in 1:total_cols
        colsize!(fig_2d.layout, col, Relative(1 / total_cols))
    end

    return fig_2d
end

###############################################################################################################
# 3. Dedicated Convergence History Plot (Train & Test Residual in the same plot)
###############################################################################################################
function plot_convergence(res_train, res_test; n_epochs_per_iter=500)
    fig = Figure(size=(850, 600), fontsize=26)
    ax = Axis(fig[1, 1],
        #title="Training & Test Convergence History",
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
        axislegend(ax, position=:rt, framevisible=true, backgroundcolor=(:white, 0.85))
    end
    =#
    return fig
end

###############################################################################################################
# 4. Deformation Modes Test (UT, PS, EB: Analytical vs Neural Network vs Training Data)
###############################################################################################################
function plot_deformation_modes(models, raw_scale, reference_function;
    x_train=nothing, y_train=nothing, train_info=nothing,
    y_train_clean=nothing, noise_level=nothing)
    mode_configs = [
        (:UT, "Uniaxial Mode (UT)", L"I_1 = \lambda^2 + 2/\lambda,\quad I_2 = 2\lambda + 1/\lambda^2", (0.6f0, 2.75f0)),
        (:PS, "Pure Shear Mode (PS)", L"I_1 = I_2 = \lambda^2 + 1 + 1/\lambda^2", (0.5f0, 2.6f0)),
        (:EB, "Equi-biaxial Mode (EB)", L"I_1 = 2\lambda^2 + 1/\lambda^4,\quad I_2 = \lambda^4 + 2/\lambda^2", (0.65f0, 1.75f0)),
    ]

    fig = Figure(size=(1650, 580), fontsize=20)

    for (col, (mode_sym, mode_title, mode_sub, default_range)) in enumerate(mode_configs)
        ax = Axis(fig[1, col],
            title=mode_title,
            subtitle=mode_sub,
            titlesize=24,
            subtitlesize=18,
            xlabel=L"\textrm{Principal Stretch }\lambda\textrm{ [-]}",
            ylabel=L"\textrm{Strain Energy Potential }\Psi\textrm{ [MPa]}",
            xgridvisible=true,
            ygridvisible=true)

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
        y_pred = vec(predict_all(models, raw_scale, x_dense))

        # Ground truth analytical and learned neural network curves
        lines!(ax, lams_dense, y_ref, color=:black, linestyle=:dash, linewidth=2.5, label=L"\textrm{Analytical }\Psi")
        lines!(ax, lams_dense, y_pred, color=:crimson, linewidth=2.8, label=L"\textrm{Network }\Psi_{\textrm{NN}}")

        # Reference line for undeformed state lambda = 1.0
        vlines!(ax, [1.0f0], color=(:gray60, 0.7), linestyle=:dot, linewidth=1.5)

        # Plot training data points for this mode if available
        if train_info !== nothing && y_train !== nothing
            mask = train_info.modes .== mode_sym
            n_pts = count(mask)
            if n_pts > 0
                lams_pts = train_info.stretches[mask]
                y_pts = vec(y_train)[mask]
                data_label = (noise_level !== nothing && noise_level > 0) ?
                             "Noisy Train Data ($(round(noise_level*100, digits=1))%)" :
                             "Train Data ($n_pts pts)"
                scatter!(ax, lams_pts, y_pts, marker=:circle, markersize=10,
                    color=(:dodgerblue2, 0.85), strokewidth=1.0, strokecolor=:navy,
                    label=data_label)
            end
        end

        if col == 1
            axislegend(ax, position=:lt, framevisible=true, backgroundcolor=(:white, 0.85))
        end
    end

    for col in 1:3
        colsize!(fig.layout, col, Relative(1 / 3))
    end

    return fig
end

# Generate and display figures
println("\nGenerating 1D Sub-network visualization...")
fig_1d = plot_1d_subnetworks(models, raw_scale, nets_x1, nets_x2, num_modes)
display(fig_1d)

println("Generating 2D Energy landscape visualization...")
fig_2d = plot_2d_energy_landscape(models, raw_scale, num_modes)
display(fig_2d)

# Save figures to plots/Visualizations/<folder_name>
save_dir_plots = plotsdir("Visualizations", folder_name)
mkpath(save_dir_plots)

path_1d = joinpath(save_dir_plots, "Polynomial_Model_Architecture_1D_Subnetworks.png")
save(path_1d, fig_1d)
printstyled("Saved 1D sub-networks figure to: $path_1d\n", color=:green, bold=true)

path_2d = joinpath(save_dir_plots, "Polynomial_Model_Architecture_2D_Energy_Landscape.png")
save(path_2d, fig_2d)
printstyled("Saved 2D energy landscape figure to: $path_2d\n", color=:green, bold=true)

if residual_over_iteration !== nothing && length(residual_over_iteration) > 0
    println("Generating Convergence History figure (Train & Test residuals)...")
    fig_conv = plot_convergence(residual_over_iteration, test_residual_over_iteration)
    display(fig_conv)
    path_conv = joinpath(save_dir_plots, "Polynomial_Convergence_History.png")
    save(path_conv, fig_conv)
    printstyled("Saved Convergence History figure to: $path_conv\n", color=:green, bold=true)
end

println("Generating Deformation Modes visualization...")
fig_modes = plot_deformation_modes(models, raw_scale, reference_function;
    x_train=x_train, y_train=y_train, train_info=train_info,
    y_train_clean=y_train_clean, noise_level=noise_level)
display(fig_modes)
path_modes = joinpath(save_dir_plots, "Incompressible_Deformation_Modes.png")
save(path_modes, fig_modes)
printstyled("Saved Deformation Modes figure to: $path_modes\n", color=:green, bold=true)