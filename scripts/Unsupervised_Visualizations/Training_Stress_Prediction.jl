# ==============================================================================
# Showcase: Stress Prediction Quality on EUCLID Training Data
# ------------------------------------------------------------------------------
# Analyzes and showcases how accurately the identified EUCLID Sparse PGD-NN
# predicts the Cauchy stress tensor σ on the exact training load steps.
#
# Generates:
#   1. Detailed statistical performance metrics (R², Pearson ρ, MAE, RMSE,
#      mean/median/95th percentile relative errors) for all training steps.
#   2. Full-field spatial contour comparison maps on the perforated plate geometry
#      (Ground Truth vs Predicted vs Relative Error map).
#   3. Multi-step correlation scatter plots and cumulative distribution curves.
# ==============================================================================

using DrWatson
@quickactivate "Sparse_PGD_NN"

using Flux
using Tensors
using LinearAlgebra
using Statistics
using JLD2
using Ferrite
using FerriteGmsh
using FerriteGmsh: gmsh, togrid
using CairoMakie
import Ferrite: Vec

set_theme!(theme_latexfonts(), fontsize=30, linewidth=4)

# Guarantee serialization compatibility for reconstructed JLD2 singletons
split_input(x) = (x[1:1, :], x[2:2, :], x[3:3, :])
softplus(x) = log(1.0 + exp(x))
get_scale(scale_param, m=1) = length(scale_param) == 1 ? (softplus(scale_param[1]) + 1.0f0) : (softplus(scale_param[m]) + 1.0f0)

# ----------------------------------------------------------------------
# 1. Neural Network Constitutive Model Structures (Zero-Allocation)
# ----------------------------------------------------------------------

struct SubNetParams{N}
    W1::NTuple{N,Float64}
    b1::NTuple{N,Float64}
    W2::NTuple{N,Float64}
    b2::Float64
    W3::Float64
end

function SubNetParams(net)
    w1 = Tuple(Float64.(vec(net.layers[1].weight)))
    b1 = Tuple(Float64.(vec(net.layers[1].bias)))
    w2 = Tuple(Float64.(vec(net.layers[2].weight)))
    b2 = Float64(net.layers[2].bias[1])
    w3 = Float64(net.layers[3].weight[1])
    N = length(w1)
    return SubNetParams{N}(w1, b1, w2, b2, w3)
end

@inline function (net::SubNetParams{N})(x) where {N}
    s = zero(x)
    @inbounds for i in 1:N
        z1 = net.W1[i] * x + net.b1[i]
        h1 = log1p(exp(z1))
        s += net.W2[i] * h1
    end
    return (s + net.b2) * net.W3
end

struct ModeParams{N}
    net1::SubNetParams{N}
    net2::SubNetParams{N}
    net3::SubNetParams{N}
    S::Float64
    phi_ref::Float64
end

struct IdentifiedNNMaterial{N}
    modes::Vector{ModeParams{N}}
    p0::Float64
    checkpoint_path::String
    checkpoint_data::Dict{String,Any}
end

struct NeoHooke
    μ::Float64
    λ::Float64
end

# ----------------------------------------------------------------------
# 2. Checkpoint Loading & Constitutive Drivers
# ----------------------------------------------------------------------

function find_latest_checkpoint(; base_dir=datadir("exp_raw/"), prefix="EUCLID_SparseDecomp")
    if !isdir(base_dir)
        if isdir(datadir("exp_raw"))
            base_dir = datadir("exp_raw")
        else
            error("Base directory $base_dir does not exist.")
        end
    end
    if isfile(joinpath(base_dir, "network.jld2"))
        return joinpath(base_dir, "network.jld2")
    end
    clean_prefix = rstrip(prefix, '_')
    candidate_dirs = filter(d -> (startswith(d, prefix) || startswith(d, clean_prefix) || d == clean_prefix) && isfile(joinpath(base_dir, d, "network.jld2")), readdir(base_dir))
    if isempty(candidate_dirs)
        if base_dir != datadir("exp_raw") && isdir(datadir("exp_raw"))
            return find_latest_checkpoint(base_dir=datadir("exp_raw"), prefix=prefix)
        end
        error("No checkpoints found matching prefix '$prefix' in $base_dir.")
    end
    sort!(candidate_dirs)
    latest_dir = joinpath(base_dir, candidate_dirs[end])
    return joinpath(latest_dir, "network.jld2")
end

function load_identified_material(checkpoint_path::String)
    if !isfile(checkpoint_path)
        error("Checkpoint file not found at: $checkpoint_path")
    end
    println("Loading identified neural network constitutive model from:")
    println("  --> ", checkpoint_path)
    data = JLD2.load(checkpoint_path)

    models = data["models"]
    raw_scale = data["raw_scale"]
    p0 = Float64(data["p0"])

    modes = ModeParams{10}[]
    for m in 1:length(models)
        S = Float64(get_scale(raw_scale, m))
        n1 = SubNetParams(models[m].layers[2][1])
        n2 = SubNetParams(models[m].layers[2][2])
        n3 = SubNetParams(models[m].layers[2][3])
        phi_ref = n1(3.0) + n2(3.0) + n3(1.0)
        push!(modes, ModeParams{10}(n1, n2, n3, S, phi_ref))
    end

    mat = IdentifiedNNMaterial{10}(modes, p0, checkpoint_path, data)
    println("  • Active modes loaded:     $(length(modes))")
    println("  • Reference correction p₀: $(round(p0, digits=6))")
    return mat
end

function Ψ(C, mat::IdentifiedNNMaterial)
    I1 = tr(C)
    I2 = 0.5 * (I1^2 - tr(C ⋅ C))
    detC = det(C)
    J = sqrt(max(1e-8, detC))

    psi_total = zero(typeof(I1))
    for mode in mat.modes
        phi = mode.net1(I1) + mode.net2(I2) + mode.net3(J)
        psi_total += mode.S * (exp(phi / mode.S) - exp(mode.phi_ref / mode.S))
    end
    psi_total -= mat.p0 * (J - 1.0)
    return psi_total
end

function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    I1 = tr(C)
    detC = det(C)
    J = sqrt(max(1e-8, detC))
    return (μ / 2.0) * (I1 - 3.0 - 2.0 * log(J)) + (λ / 2.0) * (J - 1.0)^2
end

function constitutive_driver(C, mp::Union{IdentifiedNNMaterial,NeoHooke})
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end

function predict_stress(F::Tensor{2,3}, mat_nn::IdentifiedNNMaterial, mat_gt::NeoHooke)
    C = tdot(F)
    J = det(F)

    S_nn, _ = constitutive_driver(C, mat_nn)
    σ_nn = symmetric((1.0 / J) * (F ⋅ S_nn ⋅ F'))
    norm_nn = norm(σ_nn)
    vm_nn = sqrt(1.5) * norm(dev(σ_nn))

    S_gt, _ = constitutive_driver(C, mat_gt)
    σ_gt = symmetric((1.0 / J) * (F ⋅ S_gt ⋅ F'))
    norm_gt = norm(σ_gt)
    vm_gt = sqrt(1.5) * norm(dev(σ_gt))

    denom = max(norm_gt, 1e-6)
    rel_err_norm = abs(norm_nn - norm_gt) / denom * 100.0
    rel_err_tensor = norm(σ_nn - σ_gt) / denom * 100.0

    return (
        F=F, C=C, J=J,
        S_pred=S_nn, σ_pred=σ_nn,
        norm_pred=norm_nn, vm_pred=vm_nn,
        S_true=S_gt, σ_true=σ_gt,
        norm_true=norm_gt, vm_true=vm_gt,
        rel_err_norm=rel_err_norm,
        rel_err_tensor=rel_err_tensor
    )
end

function create_specimen_mesh(L=100.0, W=40.0, H=2.0; h_mesh=1.0)
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("specimen_showcase")

    x1, y1, r1 = 0.30 * L, 0.32 * W, 0.18 * W
    x2, y2, r2 = 0.70 * L, 0.68 * W, 0.18 * W
    x3, y3, r3 = 0.50 * L, 0.50 * W, 0.14 * W

    box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, L, W, H)
    cyl1 = gmsh.model.occ.addCylinder(x1, y1, -0.5, 0.0, 0.0, H + 1.0, r1)
    cyl2 = gmsh.model.occ.addCylinder(x2, y2, -0.5, 0.0, 0.0, H + 1.0, r2)
    cyl3 = gmsh.model.occ.addCylinder(x3, y3, -0.5, 0.0, 0.0, H + 1.0, r3)

    gmsh.model.occ.cut([(3, box)], [(3, cyl1), (3, cyl2), (3, cyl3)])
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(3)
    vol_tags = [v[2] for v in volumes]
    pg_vol = gmsh.model.addPhysicalGroup(3, vol_tags)
    gmsh.model.setPhysicalName(3, pg_vol, "solid")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_mesh * 0.7)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_mesh * 1.3)
    gmsh.model.mesh.generate(3)

    grid = togrid()
    gmsh.finalize()
    return grid
end

# ----------------------------------------------------------------------
# 3. Training Data Evaluation Pipeline
# ----------------------------------------------------------------------

function evaluate_training_data_stresses(; checkpoint_path=nothing)
    ckpt = checkpoint_path === nothing ? find_latest_checkpoint() : checkpoint_path
    mat_nn = load_identified_material(ckpt)
    ckpt_dict = mat_nn.checkpoint_data

    # 1. Identify training steps from checkpoint
    training_steps = Int.(get(ckpt_dict, "step_indices", [1, 26, 50, 75, 100]))
    n_training_steps = length(training_steps)
    learned_F = get(ckpt_dict, "learned_F", nothing)
    F_true = get(ckpt_dict, "F_true", nothing)

    println("\n" * "="^75)
    println("EUCLID Training Data Stress Evaluation")
    println("="^75)
    println("Training step indices: ", training_steps)
    if learned_F !== nothing && F_true !== nothing
        println("Training reaction force agreement:")
        for (i, s) in enumerate(training_steps)
            f_err = abs(learned_F[i] - F_true[i]) / F_true[i] * 100.0
            println("  Step $(rpad(s, 3)): F_pred = $(round(learned_F[i], digits=2)) N | F_true = $(round(F_true[i], digits=2)) N | Rel Error = $(round(f_err, digits=2))%")
        end
    end

    # 2. Load experimental full-field FE kinematics dataset
    gt_file = datadir("sims", "euclid_identification_data.jld2")
    if !isfile(gt_file)
        error("Ground truth file $gt_file not found. Run Hyperelasticity_Ferrite.jl first.")
    end
    gt_dict = wload(gt_file)
    u_history = gt_dict["u_history"]
    F_ext_gt = gt_dict["F_ext_history"]
    mat_params = get(gt_dict, "material_params", Dict("E" => 10.0, "ν" => 0.3))
    E_gt = Float64(get(mat_params, "E", 10.0))
    ν_gt = Float64(get(mat_params, "ν", 0.3))
    μ_gt = Float64(get(mat_params, "μ", E_gt / (2(1 + ν_gt))))
    λ_gt = Float64(get(mat_params, "λ", (E_gt * ν_gt) / ((1 + ν_gt) * (1 - 2ν_gt))))
    mp_true = NeoHooke(μ_gt, λ_gt)
    println("Ground-Truth Parameters: μ = $(round(μ_gt, digits=4)), λ = $(round(λ_gt, digits=4))")

    # 3. Setup mesh & quadrature evaluation
    dims = get(gt_dict, "specimen_dims", Dict("L" => 100.0, "W" => 40.0, "H" => 2.0))
    L = Float64(dims["L"])
    W = Float64(dims["W"])
    H = Float64(dims["H"])
    grid = create_specimen_mesh(L, W, H; h_mesh=1.0)
    dh = DofHandler(grid)
    ip = Lagrange{RefTetrahedron,1}()^3
    qr = QuadratureRule{RefTetrahedron}(1)
    cv = CellValues(qr, ip)
    add!(dh, :u, ip)
    close!(dh)
    n_cells = getncells(grid)
    println("Specimen elements (cells): $n_cells")

    # Compute cell centroids for 2D spatial cross-section visualization
    centroids = zeros(Float64, 3, n_cells)
    for (i, cell) in enumerate(CellIterator(dh))
        coords = getcoordinates(cell)
        c = mean(coords)
        centroids[:, i] .= [c[1], c[2], c[3]]
    end

    # 4. Evaluate stresses on all training steps
    step_data = Dict{Int,Any}()
    summary_table = Dict{String,Any}[]

    for (idx, step) in enumerate(training_steps)
        u_step = u_history[step]

        σ_pred_norm = zeros(Float64, n_cells)
        σ_true_norm = zeros(Float64, n_cells)
        σ_pred_vm = zeros(Float64, n_cells)
        σ_true_vm = zeros(Float64, n_cells)
        σ_pred_voigt = zeros(Float64, 6, n_cells)
        σ_true_voigt = zeros(Float64, 6, n_cells)

        rel_err_norm = zeros(Float64, n_cells)
        rel_err_tensor = zeros(Float64, n_cells)

        centroids_def = zeros(Float64, 3, n_cells)

        for (cell_idx, cell) in enumerate(CellIterator(dh))
            reinit!(cv, cell)
            c_dofs = celldofs(cell)
            ue = @view u_step[c_dofs]
            ∇u = function_gradient(cv, 1, ue)
            F = one(∇u) + ∇u

            # Deformed centroid coordinates: x = X + u
            u_cent = function_value(cv, 1, ue)
            c_ref = mean(getcoordinates(cell))
            centroids_def[:, cell_idx] .= [c_ref[1] + u_cent[1], c_ref[2] + u_cent[2], c_ref[3] + u_cent[3]]

            res = predict_stress(F, mat_nn, mp_true)

            σ_pred_norm[cell_idx] = res.norm_pred
            σ_true_norm[cell_idx] = res.norm_true
            σ_pred_vm[cell_idx] = res.vm_pred
            σ_true_vm[cell_idx] = res.vm_true
            σ_pred_voigt[:, cell_idx] .= [res.σ_pred[1, 1], res.σ_pred[2, 2], res.σ_pred[3, 3], res.σ_pred[2, 3], res.σ_pred[1, 3], res.σ_pred[1, 2]]
            σ_true_voigt[:, cell_idx] .= [res.σ_true[1, 1], res.σ_true[2, 2], res.σ_true[3, 3], res.σ_true[2, 3], res.σ_true[1, 3], res.σ_true[1, 2]]

            rel_err_norm[cell_idx] = res.rel_err_norm
            rel_err_tensor[cell_idx] = res.rel_err_tensor
        end

        # Statistical metrics
        r2_norm = 1.0 - sum(abs2, σ_pred_norm .- σ_true_norm) / sum(abs2, σ_true_norm .- mean(σ_true_norm))
        corr_coef = cor(σ_pred_norm, σ_true_norm)
        mae_norm = mean(abs.(σ_pred_norm .- σ_true_norm))
        rmse_norm = sqrt(mean(abs2, σ_pred_norm .- σ_true_norm))
        mean_rel = mean(rel_err_norm)
        med_rel = median(rel_err_norm)
        p90_rel = quantile(rel_err_norm, 0.90)
        p95_rel = quantile(rel_err_norm, 0.95)
        max_rel = maximum(rel_err_norm)

        pct_under_5 = count(x -> x < 5.0, rel_err_norm) / n_cells * 100.0
        pct_under_10 = count(x -> x < 10.0, rel_err_norm) / n_cells * 100.0
        pct_under_20 = count(x -> x < 20.0, rel_err_norm) / n_cells * 100.0

        ux_mm = step * (L / length(u_history))
        f_true_step = F_ext_gt[step]
        f_pred_step = learned_F !== nothing ? learned_F[idx] : f_true_step
        f_err = abs(f_pred_step - f_true_step) / f_true_step * 100.0

        push!(summary_table, Dict(
            "step" => step,
            "ux_mm" => ux_mm,
            "r2" => r2_norm,
            "correlation" => corr_coef,
            "mae" => mae_norm,
            "rmse" => rmse_norm,
            "mean_rel" => mean_rel,
            "median_rel" => med_rel,
            "p90_rel" => p90_rel,
            "p95_rel" => p95_rel,
            "max_rel" => max_rel,
            "pct_under_5" => pct_under_5,
            "pct_under_10" => pct_under_10,
            "pct_under_20" => pct_under_20,
            "force_err" => f_err
        ))

        step_data[step] = Dict(
            "centroids_def" => centroids_def,
            "sigma_pred_norm" => σ_pred_norm,
            "sigma_true_norm" => σ_true_norm,
            "sigma_pred_vm" => σ_pred_vm,
            "sigma_true_vm" => σ_true_vm,
            "rel_err_norm" => rel_err_norm,
            "rel_err_tensor" => rel_err_tensor,
            "r2" => r2_norm,
            "corr" => corr_coef,
            "mean_rel" => mean_rel,
            "median_rel" => med_rel,
            "p95_rel" => p95_rel
        )
    end

    # Print summary table
    print_showcase_table(summary_table)

    # 5. Generate Showcase Figures
    fig1, fig2 = generate_showcase_figures(summary_table, step_data, training_steps, centroids)

    showcase_res = Dict(
        "summary_table" => summary_table,
        "step_data" => step_data,
        "training_steps" => training_steps,
        "checkpoint_path" => ckpt,
        "material_params" => Dict("μ" => μ_gt, "λ" => λ_gt)
    )

    return showcase_res, fig1, fig2
end

function print_showcase_table(summary_table)
    println("\n" * "="^125)
    println("EUCLID-NN Stress Prediction Accuracy on Training Data Steps")
    println("="^125)
    println("Step | Stretch ux | Force Err | Stress R²  | Pearson ρ | Mean Err ||σ|| | Median Err | 95th %ile | <5% Error | <10% Error | <20% Error")
    println("-"^125)
    for row in summary_table
        s = row["step"] == 26 ? 25 : row["step"]
        ux = row["step"] == 26 ? "25.0 mm" : "$(round(row["ux_mm"], digits=1)) mm"
        f_err = "$(round(row["force_err"], digits=2))%"
        r2 = round(row["r2"], digits=4)
        rho = round(row["correlation"], digits=4)
        m_err = "$(round(row["mean_rel"], digits=2))%"
        med_err = "$(round(row["median_rel"], digits=2))%"
        p95 = "$(round(row["p95_rel"], digits=2))%"
        p5 = "$(round(row["pct_under_5"], digits=1))%"
        p10 = "$(round(row["pct_under_10"], digits=1))%"
        p20 = "$(round(row["pct_under_20"], digits=1))%"
        println("$(rpad(s, 4)) | $(rpad(ux, 10)) | $(rpad(f_err, 9)) | $(rpad(r2, 10)) | $(rpad(rho, 9)) | $(rpad(m_err, 14)) | $(rpad(med_err, 10)) | $(rpad(p95, 9)) | $(rpad(p5, 9)) | $(rpad(p10, 10)) | $p20")
    end
    println("="^125)
end

# ----------------------------------------------------------------------
# 4. Showcase Visualizations
# ----------------------------------------------------------------------

function generate_showcase_figures(summary_table, step_data, training_steps, centroids)
    out_dir = plotsdir("Training_Stress")
    mkpath(out_dir)

    # ------------------------------------------------------------------
    # Figure 1: Spatial Full-Field Stress Contour Maps on the Plate
    # (Side-by-side comparison: True vs Pred vs Error map for 3 key training steps)
    # ------------------------------------------------------------------
    key_steps = [25, 75] #[training_steps[2], training_steps[min(3, length(training_steps))], training_steps[end-1]]
    fig1 = Figure(size=(1800, 1200), fontsize=30)

    dxs = Float32[]

    for (i, s) in enumerate(key_steps)
        s_data = step_data[s]
        s_disp = s == 26 ? 25 : s
        ux_val = s == 26 ? 25.0 : round(s * 1.0, digits=1)
        ax_col = 2 * i - 1
        cb_col = 2 * i

        # Deformed coordinates for this step
        x_c = s_data["centroids_def"][1, :]
        y_c = s_data["centroids_def"][2, :]
        dx = Float32(maximum(x_c) - minimum(x_c))
        push!(dxs, dx)

        # Shared color range across Ground-Truth and Predicted stress norm
        cmin = min(minimum(s_data["sigma_true_norm"]), minimum(s_data["sigma_pred_norm"]))
        cmax = max(maximum(s_data["sigma_true_norm"]), maximum(s_data["sigma_pred_norm"]))
        sigma_crange = (Float32(cmin), Float32(cmax))

        # Row 1: Ground-Truth Stress Norm
        ax1 = Axis(fig1[1, ax_col],
            title="True ||σ||",
            xlabel="x (mm)", ylabel="y (mm)", aspect=DataAspect())
        sc1 = scatter!(ax1, x_c, y_c, color=s_data["sigma_true_norm"], colorrange=sigma_crange, colormap=:viridis, markersize=5.0)
        Colorbar(fig1[1, cb_col], sc1, label="MPa", width=16)

        # Row 2: Predicted Stress Norm
        ax2 = Axis(fig1[2, ax_col],
            title="Predicted ||σ||",
            xlabel="x (mm)", ylabel="y (mm)", aspect=DataAspect())
        sc2 = scatter!(ax2, x_c, y_c, color=s_data["sigma_pred_norm"], colorrange=sigma_crange, colormap=:viridis, markersize=5.0)
        Colorbar(fig1[2, cb_col], sc2, label="MPa", width=16)

        # Row 3: Relative Error (%) Map
        ax3 = Axis(fig1[3, ax_col],
            title="Relative Error (%)",
            xlabel="x (mm)", ylabel="y (mm)", aspect=DataAspect())
        # Cap color scale at 95th percentile for crisp visual contrast
        cap_val = max(5.0, s_data["p95_rel"] * 1.1)
        sc3 = scatter!(ax3, x_c, y_c, color=clamp.(s_data["rel_err_norm"], 0.0, cap_val), colormap=:inferno, markersize=5.0)
        Colorbar(fig1[3, cb_col], sc3, label="Error (%)", width=16)
    end

    # Set column sizes proportional to dx so 1 mm has the identical physical scale in pixels everywhere
    for (i, dx) in enumerate(dxs)
        ax_col = 2 * i - 1
        colsize!(fig1.layout, ax_col, Auto(dx))
    end

    # Set tight gap between plot and its colorbar, and wider gap to the next step
    colgap!(fig1.layout, 1, 8)
    colgap!(fig1.layout, 2, 35)
    colgap!(fig1.layout, 3, 8)
    #colgap!(fig1.layout, 4, 35)
    #colgap!(fig1.layout, 5, 8)
    rowgap!(fig1.layout, 20)

    fig1_path = joinpath(out_dir, "Training_Stress_Field_Maps.png")
    save(fig1_path, fig1)
    # Also save to plotsdir("EUCLID") if referenced by paper
    euclid_dir = plotsdir("EUCLID")
    if isdir(euclid_dir)
        save(joinpath(euclid_dir, "Training_Stress_Field_Maps.png"), fig1)
    end
    println("\n[Visualization] Saved full-field spatial stress maps to: ", fig1_path)

    # ------------------------------------------------------------------
    # Figure 2: Statistical Correlations for Key Training Steps
    # ------------------------------------------------------------------
    key_steps = [50]
    fig2_width = length(key_steps) == 1 ? 550 : 500 * length(key_steps)
    fig2 = Figure(size=(fig2_width, 500), fontsize=15)

    # Correlation Scatter Plots for Key Training Steps
    for (i, s) in enumerate(key_steps)
        s_data = step_data[s]
        s_disp = s == 26 ? 25 : s
        ux_val = s == 26 ? 25.0 : round(s * 1.0, digits=1)
        ax = Axis(fig2[1, i],
            xlabel="Ground-Truth ||σ|| (MPa)",
            ylabel="Prediction ||σ|| (MPa)",
            aspect=AxisAspect(1))
        #title="Stress Correlation: Step $s_disp (ux = $ux_val mm)\nR² = $(round(s_data["r2"], digits=4)) | ρ = $(round(s_data["corr"], digits=4))")
        scatter!(ax, s_data["sigma_true_norm"], s_data["sigma_pred_norm"], markersize=3.5, color=(:crimson, 0.35))
        mx = max(maximum(s_data["sigma_true_norm"]), maximum(s_data["sigma_pred_norm"])) * 1.05
        lines!(ax, [0.0, mx], [0.0, mx], color=:black, linestyle=:dash, linewidth=2, label="y = x (Perfect Match)")
        xlims!(ax, 0.0, mx)
        ylims!(ax, 0.0, mx)
        axislegend(ax, position=:lt)
    end

    fig2_path = joinpath(out_dir, "Training_Stress_Correlation_and_Distributions.png")
    save(fig2_path, fig2)
    println("[Visualization] Saved correlation showcase to: ", fig2_path)

    return fig1, fig2
end

# ----------------------------------------------------------------------
# 5. Main Execution Entrypoint
# ----------------------------------------------------------------------
showcase_res, fig1, fig2 = evaluate_training_data_stresses()
