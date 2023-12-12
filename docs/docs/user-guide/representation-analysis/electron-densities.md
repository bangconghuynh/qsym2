---
title: Electron densities
description: Configurable parameters for electron density representation analysis
---

# Electron densities

Let $\rho(\mathbfit{r})$ be the one-electron density defined based on an $N_{\mathrm{e}}$-electron wavefunction $\Psi(\mathbfit{x}_1, \ldots, \mathbfit{x}_{N_{\mathrm{e}}})$ as

$$
    \rho(\mathbfit{r}) =
        N_{\mathrm{e}} \int
            \Psi(\mathbfit{r}, s, \mathbfit{x}_2, \ldots, \mathbfit{x}_{N_{\mathrm{e}}})^*
            \Psi(\mathbfit{r}, s, \mathbfit{x}_2, \ldots, \mathbfit{x}_{N_{\mathrm{e}}})
            \ \mathrm{d}s \ \mathrm{d}\mathbfit{x}_2 \ldots \mathrm{d}\mathbfit{x}_{N_{\mathrm{e}}}.
$$

In an atomic-orbital basis $\{ \phi_{\gamma}(\mathbfit{r}), \phi_{\delta}(\mathbfit{r}), \ldots \}$, the density $\rho(\mathbfit{r})$ can be expanded as

$$
    \rho(\mathbfit{r}) = \sum_{\gamma \delta}
        \phi_{\gamma}(\mathbfit{r}) \phi_{\delta}(\mathbfit{r}) P_{\delta \gamma},
$$

where $P_{\delta \gamma}$ are elements of the corresponding density matrix $\mathbfit{P}$ in this basis.
QSym² is able to provide symmetry assignments for electron densities and derived quantities &mdash; the mathematical details of this can be found in [Section 2.4.2 of the QSym² paper](../../about/authorship.md#publications).


## Requirements

### Basis overlap matrix

As explained in [Section 2.4.2 of the QSym² paper](../../about/authorship.md#publications), the symmetry analysis of electron densities requires four-centre overlap integrals:

$$
    \braket{\phi_{\gamma} \phi_{\delta} | \phi_{\gamma'} \phi_{\delta'}} = \int
        \phi_{\gamma}^*(\mathbfit{r}) \phi_{\delta}^*(\mathbfit{r})
        \phi_{\gamma'}(\mathbfit{r}) \phi_{\delta'}(\mathbfit{r})
        \ \mathrm{d}\mathbfit{r}.
$$

Few programs are known to have these integrals computed as part of their routine calculations.
QSym² therefore has implementations to calculate these integrals, 
