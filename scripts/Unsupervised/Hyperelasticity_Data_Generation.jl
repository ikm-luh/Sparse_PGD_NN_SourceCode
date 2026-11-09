# ==============================================================================
# Unsupervised Data Generation: 3D FE Perforated Plate Tensile Test
# ------------------------------------------------------------------------------
# Generates full-field synthetic experimental data for the unsupervised
# constitutive law discovery framework (EUCLID / Sparse PGD-NN).
#
# Methodology:
#   1. Geometry & Mesh: Perforated 3D plate with three offset holes generated
#      via FerriteGmsh. The cutouts create heterogeneous, multiaxial strain fields
#      spanning tension, compression, and shear in a single experiment.
#   2. Nonlinear FE Solver: Solves quasi-static finite strain hyperelasticity
#      using Ferrite.jl and Tensors.jl with Newton-Raphson iteration.
#   3. Virtual Work Operators: Computes pre-assembled sparse virtual work basis
#      matrices (K₁, K₂, K₃) at each load step such that:
#          g_int = K₁ * (∂Ψ/∂I₁) + K₂ * (∂Ψ/∂I₂) + K₃ * (∂Ψ/∂J)
#      enabling efficient, fast automatic differentiation during neural network training.
#   4. Output: Saves `euclid_identification_data.jld2` to `data/sims/`.
# ==============================================================================

using DrWatson
@quickactivate "Sparse_PGD_NN"

using Flux
using CairoMakie
using Statistics
using Random
using LinearAlgebra
using Dates

using Ferrite, Tensors, TimerOutputs, ProgressMeter, IterativeSolvers
using FerriteGmsh
using FerriteGmsh: gmsh, togrid
using SparseArrays
using JLD2
import Ferrite: Vec

"""
Compressible Neo-Hookean ground-truth material model:
    Ψ(C) = (μ/2) * (I_c - 3 - 2*ln(J)) + (λ/2) * (J - 1)²
"""
struct NeoHooke
    μ::Float64
    λ::Float64
end
function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end
function constitutive_driver(C, mp::NeoHooke)
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;
function assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN)
    # Reinitialize cell values
    reinit!(cv, cell)
    b = zero(Vec{3}) # Body force (negligible in material identification tensile tests)
    tn = 0.0 # Traction on free boundaries
    ndofs = getnbasefunctions(cv)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F) # F' ⋅ F
        # Compute stress and tangent
        S, ∂S∂C = constitutive_driver(C, mp)
        P = F ⋅ S
        I = one(S)
        ∂P∂F = otimesu(I, S) + 2 * F ⋅ ∂S∂C ⊡ otimesu(F', I)
        # Loop over test functions
        for i in 1:ndofs
            # Test function and gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            # Add contribution to the residual from this test function
            ge[i] += (∇δui ⊡ P - δui ⋅ b) * dΩ
            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end
    # Surface integral for the traction
    for facet in 1:nfacets(cell)
        if (cellid(cell), facet) in ΓN
            reinit!(fv, cell, facet)
            for q_point in 1:getnquadpoints(fv)
                t = tn * getnormal(fv, q_point)
                dΓ = getdetJdV(fv, q_point)
                for i in 1:ndofs
                    δui = shape_value(fv, q_point, i)
                    ge[i] -= (δui ⋅ t) * dΓ
                end
            end
        end
    end
    return
end;
function assemble_global!(K, g, dh, cv, fv, mp, u, ΓN)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)
    ue = zeros(n)
    # start_assemble resets K and g
    assembler = start_assemble(K, g)
    # Loop over all cells in the grid
    @timeit "assemble" for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue .= @view u[global_dofs] # element dofs
        fill!(ke, 0.0)
        fill!(ge, 0.0)
        @timeit "element assemble" assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN)
        assemble!(assembler, global_dofs, ke, ge)
    end
    return
end;
"""
    generate_perforated_plate_grid(; L=100.0, W=40.0, H=2.0, h_mesh=4.0)

Generate a 3D thin plate with two offset circular holes using Gmsh (FerriteGmsh),
as typically used in experimental parameter identification of hyperelastic materials
(e.g., DIC tensile tests creating heterogeneous multiaxial strain fields).
"""
function generate_perforated_plate_grid(; L=100.0, W=40.0, H=2.0, h_mesh=4.0)
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("perforated_plate")

    # Offset holes produce inhomogeneous tension, shear, and compression zones
    # Hole 1:
    x1, y1, r1 = 0.30 * L, 0.32 * W, 0.18 * W
    # Hole 2:
    x2, y2, r2 = 0.70 * L, 0.68 * W, 0.18 * W
    # Hole 3 (central staggered cutout for smooth gradient transition):
    x3, y3, r3 = 0.50 * L, 0.50 * W, 0.14 * W

    box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, L, W, H)
    cyl1 = gmsh.model.occ.addCylinder(x1, y1, -0.5, 0.0, 0.0, H + 1.0, r1)
    cyl2 = gmsh.model.occ.addCylinder(x2, y2, -0.5, 0.0, 0.0, H + 1.0, r2)
    cyl3 = gmsh.model.occ.addCylinder(x3, y3, -0.5, 0.0, 0.0, H + 1.0, r3)

    gmsh.model.occ.cut([(3, box)], [(3, cyl1), (3, cyl2), (3, cyl3)])
    gmsh.model.occ.synchronize()

    # Classify boundary surfaces
    surfaces = gmsh.model.getEntities(2)
    left_surfs = Int32[]
    right_surfs = Int32[]
    free_surfs = Int32[]

    eps = 1e-3
    for (_, s) in surfaces
        bb = gmsh.model.getBoundingBox(2, s)
        if abs(bb[1]) < eps && abs(bb[4]) < eps
            push!(left_surfs, s)
        elseif abs(bb[1] - L) < eps && abs(bb[4] - L) < eps
            push!(right_surfs, s)
        else
            push!(free_surfs, s)
        end
    end

    pg_left = gmsh.model.addPhysicalGroup(2, left_surfs)
    gmsh.model.setPhysicalName(2, pg_left, "left")

    pg_right = gmsh.model.addPhysicalGroup(2, right_surfs)
    gmsh.model.setPhysicalName(2, pg_right, "right")

    if !isempty(free_surfs)
        pg_free = gmsh.model.addPhysicalGroup(2, free_surfs)
        gmsh.model.setPhysicalName(2, pg_free, "free")
    end

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

function solve()
    reset_timer!()
    # Generate thin plate with two holes via FerriteGmsh
    L = 100.0
    W = 40.0
    H = 2.0
    grid = @timeit "generate mesh" generate_perforated_plate_grid(; L=L, W=W, H=H, h_mesh=1.0)

    # Material parameters
    E = 10.0
    ν = 0.3
    μ = 4.0 #E / (2(1 + ν))
    λ = 6.0 #(E * ν) / ((1 + ν) * (1 - 2ν))
    mp = NeoHooke(μ, λ)

    # Finite element base
    ip = Lagrange{RefTetrahedron,1}()^3
    qr = QuadratureRule{RefTetrahedron}(1)
    qr_facet = FacetQuadratureRule{RefTetrahedron}(1)
    cv = CellValues(qr, ip)
    fv = FacetValues(qr_facet, ip)

    # DofHandler
    dh = DofHandler(grid)
    add!(dh, :u, ip) # Add a displacement field
    close!(dh)

    # Boundary conditions typical of material identification tensile test:
    # Clamped on the left edge (x = 0)
    # Tensile elongation on the right edge (x = L)
    Δu_x = 100.0 # 100 mm tensile stretch (100% engineering strain)
    dbcs = ConstraintHandler(dh)
    dbc_left = Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(dbcs, dbc_left)
    dbc_right = Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [t * Δu_x, 0.0, 0.0], [1, 2, 3])
    add!(dbcs, dbc_right)
    close!(dbcs)

    # Dedicated constraint handler to isolate pull DOFs in x direction
    ch_pull = ConstraintHandler(dh)
    dbc_right_x = Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [1.0], [1])
    add!(ch_pull, dbc_right_x)
    close!(ch_pull)
    pull_dofs_x = copy(ch_pull.prescribed_dofs)
    free_dofs_list = free_dofs(dbcs)

    # Traction-free boundaries (free surfaces and holes)
    ΓN = Set{FacetIndex}()

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    u = zeros(_ndofs)
    Δu = zeros(_ndofs)
    ΔΔu = zeros(_ndofs)

    # Create sparse matrix and residual vector
    K = allocate_matrix(dh)
    g = zeros(_ndofs)

    NEWTON_TOL = 1.0e-7
    NEWTON_MAXITER = 30
    n_steps = 100
    n_cells = getncells(grid)

    # Storage for EUCLID unsupervised identification dataset
    u_history = Vector{Vector{Float64}}()
    F_ext_history = Float64[]
    K1_history = Vector{SparseMatrixCSC{Float64,Int}}()
    K2_history = Vector{SparseMatrixCSC{Float64,Int}}()
    K3_history = Vector{SparseMatrixCSC{Float64,Int}}()
    I1_history = Vector{Vector{Float32}}()
    I2_history = Vector{Vector{Float32}}()
    J_history = Vector{Vector{Float32}}()

    @timeit "solve steps" for step in 1:n_steps
        t = step / n_steps
        Ferrite.update!(dbcs, t)
        apply!(u, dbcs)
        fill!(Δu, 0.0)

        prog = ProgressMeter.ProgressThresh(NEWTON_TOL; desc="Step $step/$n_steps (t=$t):")
        newton_itr = 0
        while true
            newton_itr += 1
            @timeit "assemble" assemble_global!(K, g, dh, cv, fv, mp, u .+ Δu, ΓN)
            apply_zero!(K, g, dbcs)

            normg = norm(g)
            ProgressMeter.update!(prog, normg; showvalues=[(:step, step), (:iter, newton_itr)])
            if normg < NEWTON_TOL
                u .+= Δu
                break
            elseif newton_itr >= NEWTON_MAXITER
                error("Reached maximum Newton iterations at step $step, aborting")
            end

            # Linear solve
            @timeit "linear solve" begin
                ΔΔu .= K \ g
            end
            apply_zero!(ΔΔu, dbcs)
            Δu .-= ΔΔu
        end

        # Compute unconstrained internal force vector g at converged state
        assemble_global!(K, g, dh, cv, fv, mp, u, ΓN)
        F_react = sum(g[pull_dofs_x])

        push!(u_history, copy(u))
        push!(F_ext_history, F_react)

        # Precompute virtual work operators K1, K2, K3 and kinematics for this step
        rows = Int[]
        cols = Int[]
        vals_K1 = Float64[]
        vals_K2 = Float64[]
        vals_K3 = Float64[]

        I1_step = zeros(Float32, n_cells)
        I2_step = zeros(Float32, n_cells)
        J_step = zeros(Float32, n_cells)

        for (cell_idx, cell) in enumerate(CellIterator(dh))
            reinit!(cv, cell)
            dΩ = getdetJdV(cv, 1)
            ue = @view u[celldofs(cell)]
            ∇u = function_gradient(cv, 1, ue)
            F = one(∇u) + ∇u
            C = tdot(F)
            I_c = tr(C)
            detC = det(C)
            J_val = sqrt(detC)
            C_inv = inv(C)
            I_tensor = one(C)

            I1_step[cell_idx] = Float32(I_c)
            I2_step[cell_idx] = Float32(0.5 * (I_c^2 - tr(C ⋅ C)))
            J_step[cell_idx] = Float32(J_val)

            # Virtual work basis tensors for hyperelastic invariants:
            # First Piola-Kirchhoff stress P = 2 * F * (∂Ψ/∂C) decomposes via chain rule as:
            #   P = (∂Ψ/∂I₁) * A₁ + (∂Ψ/∂I₂) * A₂ + (∂Ψ/∂J) * A₃
            # where:
            #   A₁ = 2 * F * (∂I₁/∂C) = 2 * F * I = 2 * F
            #   A₂ = 2 * F * (∂I₂/∂C) = 2 * F * (I_c * I - C)
            #   A₃ = 2 * F * (∂J/∂C)  = 2 * F * (0.5 * J * C⁻¹) = F * (J * C⁻¹)
            # This allows precomputing the spatial virtual work operator matrices K_k:
            #   (g_int)_i = ∑_cells [ K_{1,ic}*(∂Ψ/∂I₁) + K_{2,ic}*(∂Ψ/∂I₂) + K_{3,ic}*(∂Ψ/∂J) ]
            A1 = 2.0 * F
            A2 = 2.0 * F ⋅ (I_c * I_tensor - C)
            A3 = F ⋅ (J_val * C_inv)

            c_dofs = celldofs(cell)
            # 4 nodes per linear tetrahedral element x 3 DOFs per node = 12 element DOFs
            for i in 1:12
                ∇δui = shape_gradient(cv, 1, i)
                B_i = dΩ * ∇δui
                push!(rows, c_dofs[i])
                push!(cols, cell_idx)
                push!(vals_K1, B_i ⊡ A1)
                push!(vals_K2, B_i ⊡ A2)
                push!(vals_K3, B_i ⊡ A3)
            end
        end

        push!(K1_history, sparse(rows, cols, vals_K1, _ndofs, n_cells))
        push!(K2_history, sparse(rows, cols, vals_K2, _ndofs, n_cells))
        push!(K3_history, sparse(rows, cols, vals_K3, _ndofs, n_cells))
        push!(I1_history, I1_step)
        push!(I2_history, I2_step)
        push!(J_history, J_step)
    end

    # Export EUCLID unsupervised identification dataset
    export_dict = Dict(
        "n_cells" => n_cells,
        "ndofs" => _ndofs,
        "n_steps" => n_steps,
        "free_dofs" => free_dofs_list,
        "pull_dofs_x" => pull_dofs_x,
        "F_ext_history" => F_ext_history,
        "u_history" => u_history,
        "K1_history" => K1_history,
        "K2_history" => K2_history,
        "K3_history" => K3_history,
        "I1_history" => I1_history,
        "I2_history" => I2_history,
        "J_history" => J_history,
        "material_params" => Dict("E" => E, "ν" => ν, "μ" => μ, "λ" => λ),
        "specimen_dims" => Dict("L" => L, "W" => W, "H" => H)
    )
    save_path = datadir("sims", "euclid_identification_data.jld2")
    wsave(save_path, export_dict)
    println("\n[EUCLID Export] Identification dataset saved successfully to: ", save_path)

    # Save the solution to VTK (including nodal displacements and cell invariants)
    @timeit "export" begin
        VTKGridFile(datadir("hyperelasticity"), dh) do vtk
            write_solution(vtk, dh, u)
            write_cell_data(vtk, I1_history[end], "I1")
            write_cell_data(vtk, I2_history[end], "I2")
            write_cell_data(vtk, J_history[end], "J")
        end
    end
    print_timer(title="Analysis with $(getncells(grid)) elements", linechars=:ascii)
    return u, F_ext_history
end
u, F_ext_history = solve();