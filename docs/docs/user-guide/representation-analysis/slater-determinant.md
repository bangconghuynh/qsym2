---
title: Slater determinant
description: Configurable parameters for Slater determinant representation analysis
---

# Slater determinant

Let $\Psi_{\mathrm{SD}}$ be an $N_{\mathrm{e}}$-electron Slater determinant constructed from $N_{\mathrm{e}}$ occupied spin-orbitals $\chi_i(\mathbfit{x})$ written in terms of the composite spin-spatial coordinates $\mathbfit{x}$:

$$
    \Psi_{\mathrm{SD}} =
        \sqrt{N_{\mathrm{e}}!} \hat{\mathscr{A}}
        \left[ \prod_{i=1}^{N_{\mathrm{e}}} \chi_i(\mathbfit{x}_i) \right],
$$

where $\hat{\mathscr{A}}$ is the antisymmetriser in the symmetric group $\operatorname{Sym}(N_{\mathrm{e}})$ acting on the electron labels.
The Slater determinant $\Psi_{\mathrm{SD}}$ exists in some subspace of the $N_{\mathrm{e}}$-electron Hilbert space $\mathcal{H}_{N_{\mathrm{e}}}$ whilst the spin-orbitals $\chi_i(\mathbfit{x})$ each belong to some subspace of the one-electron Hilbert space $\mathcal{H}_{1}$.
QSym² is able to provide symmetry assignments for both the Slater determinant $\Psi_{\mathrm{SD}}$ and its constituting spin-orbitals $\chi_i(\mathbfit{x})$.

## Requirements

### Basis overlap matrix

As explained [here](basic-requirements.md/#basis-overlap-matrix), QSym² requires the overlap matrices of the bases chosen for (some subspaces of) $\mathcal{H}_{N_{\mathrm{e}}}$ and $\mathcal{H}_{1}$ in order to perform representation analysis on $\Psi_{\mathrm{SD}}$ and $\chi_i(\mathbfit{x})$, respectively.
As it turns out, since $\Psi_{\mathrm{SD}}$ is constructed from $\chi_i(\mathbfit{x})$, QSym² only requires the overlap matrix for the basis functions on $\mathcal{H}_{1}$ with respect to which the spin-orbitals $\chi_i(\mathbfit{x})$ are defined.
These basis functions are typically Gaussian atomic orbitals, and most, if not all, quantum-chemistry packages compute their overlaps as part of their inner working.
It is therefore more convenient to retrieve the atomic-orbital overlap matrix $\mathbfit{S}_{\mathcal{H}_{1}}$ from quantum-chemistry packages whenever possible.
