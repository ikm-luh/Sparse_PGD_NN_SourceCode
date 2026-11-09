# Sparse_PGD_NN

Accompanying code for the paper "Sparse Constitutive Neural Networks for Hyperelasticity".

Abstract:
Constitutive model identification aims to recover compact, interpretable models that accurately describe material behavior. In this work, we identify hyperelastic constitutive models using artificial neural networks with sparsity-promoting constraints and regularization. The proposed method combines the expressiveness of neural networks with the structural interpretability of sparse constitutive models, such as those identified by EUCLID, by representing the strain energy \revv{density} as a sum of products of univariate functions of the invariants.
The resulting decomposition exposes the contributions of individual invariants and their interactions.
The method is demonstrated on synthetic data in supervised and unsupervised learning scenarios, showing that it can accurately identify interpretable constitutive models with few active modes.

The code is located in the scripts folder and separated into Supervised and Unsupervised Learning.
The Visualizations are in their own folders named "Supervised_Visualizations" and "Unsupervised_Visualizations".

Get the code running: 
This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> Sparse_PGD_NN

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "Sparse_PGD_NN"
```
which auto-activate the project and enable local path handling from DrWatson.

Funded by the European Union (ERC, Gen-TSM, project number 101124463). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.