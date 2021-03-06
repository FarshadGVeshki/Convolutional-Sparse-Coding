# Convolutional-Sparse-Coding

(Last update 9.4.2022: stopping criteria)

Run the demo files.

The codes include:

1) unconstraned CSC algorithm: CSC_unconstrained.m
2) constraned CSC algorithm: CSC_constrained.m
3) the consensus ADMM-based CDL method: CDL.m
4) the consensus ADMM-based multiscale CDL method: CDL_multiscale.m
5) the ADMM-based CDL method based on direct matrix inversion: CDL_mtx_inv.m
6) code for generating Gaussian random multiscale dictionaries: initdict.m
7) code for visualizing multiscale filters: dict2image.m
8) pre-learned dictionaries (.mat files)

Training images are collected from USC-SIPI database.

Reference : F. G. Veshki and S. A. Vorobyov, "Efficient ADMM-based Algorithms for Convolutional Sparse Coding," in IEEE Signal Processing Letters, doi: 10.1109/LSP.2021.3135196.
Email: farshad.ghorbaniveshki@aalto.fi
