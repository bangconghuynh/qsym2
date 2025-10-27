---
title: Basics
description: Basic information for representation analysis
---

# Basics

!!! warning

    Unless stated otherwise, $\mathcal{G}$ denotes a *unitary-represented* symmetry group, which can be a unitary symmetry group or a magnetic symmetry group in which antiunitary operators are represented unitarily.
    Projection operators for magnetic-represented groups are not yet supported.

Let $V$ be a linear space and $\mathbfit{w} \in V$ be an element to be projected by QSym² onto a particular irreducible representation $\Gamma$ of a *unitary* group $\mathcal{G}$.
This amounts to constructing the orbit

$$
    \mathcal{G} \cdot \mathbfit{w} = \{ \hat{g}_i \mathbfit{w} \ :\ g_i \in \mathcal{G} \}
$$

and then computing the sum

$$
    \hat{\mathscr{P}}^{(\Gamma)} \mathbfit{w} = \frac{d_{\Gamma}}{|\mathcal{G}|} \sum_{i = 1}^{|\mathcal{G}|} \chi^{(\Gamma)}(g_i)^* (\hat{g}_i \mathbfit{w}),
$$

where $d_{\Gamma}$ is the dimension of the irreducible representation $\Gamma$ and $\chi^{(\Gamma)}$ its character function.

To perform symmetry projection in QSym², the [atomic-orbital basis angular order](../representation-analysis/basics.md#atomic-orbital-basis-angular-order) information for the underlying electronic-structure calculation is required.
