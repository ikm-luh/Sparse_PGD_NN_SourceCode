# ==============================================================================
# Sparse_PGD_NN: Startup & Environment Initialization
# ------------------------------------------------------------------------------
# Accompanying code for the paper:
# "Sparse Constitutive Neural Networks for Hyperelasticity"
#
# Usage:
#   1. Open Julia in the project root:
#      julia --project=.
#   2. Or in an active Julia REPL, run:
#      using DrWatson
#      @quickactivate "Sparse_PGD_NN"
#
# Available Workflows:
#   - Supervised Learning:
#       scripts/Supervised/Polynomial.jl   (Incompressible polynomial / Mooney-Rivlin)
#       scripts/Supervised/NeoHooke.jl      (Compressible Neo-Hookean)
#   - Unsupervised Learning (EUCLID):
#       scripts/Unsupervised/Hyperelasticity_Data_Generation.jl  (FE simulation & K-matrix generation)
#       scripts/Unsupervised/Identification_SparseDecomp.jl     (Unsupervised identification)
#   - Visualizations:
#       scripts/Supervised_Visualizations/...
#       scripts/Unsupervised_Visualizations/...
# ==============================================================================

using DrWatson
@quickactivate "Sparse_PGD_NN"

println("Sparse_PGD_NN project activated successfully.")
println("Project directory: ", projectdir())
