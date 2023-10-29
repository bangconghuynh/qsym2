---
title: Development
description: Development of QSym²
---

# Development

## Authors

QSym² (pronounced *q-sym-squared*) has been developed and maintained by Dr [Bang C. Huynh](https://orcid.org/0000-0002-5226-4054) at the University of Nottingham, UK since July 2022 with scientific support from Prof. [Andrew M. Wibowo-Teale](https://orcid.org/0000-0001-9617-1143) and Dr [Meilani Wibowo-Teale](https://orcid.org/0000-0003-2462-3328) and financial support from the ERC grant under the *topDFT* project.

The logo for QSym², which is a stylised stellated octahedron, was designed with artistic support from Mr [Thinh Nguyen](https://www.linkedin.com/in/thinh-nguyen-a38b7856/).

## Publications

The core functionalities of QSym² are described and illustrated in the following manuscript:

- Huynh, B. C., Wibowo-Teale, M. & Wibowo-Teale, A. M. QSym²: A Quantum Symbolic Symmetry Analysis Program for Electronic Structure. Preprint at [arXiv:2310.06749](http://arxiv.org/abs/2310.06749) (2023).

Research work utilising QSym² for symmetry analysis can be found in the following select publications:

- Wibowo, M., Huynh, B. C., Cheng, C. Y., Irons, T. J. P. & Teale, A. M. Understanding ground and excited-state molecular structure in strong magnetic fields using the maximum overlap method. *Molecular Physics* **121**, e2152748 (2023), [DOI](https://doi.org/10.1080/00268976.2022.2152748).
- Wibowo-Teale, M., Ennifer, B. J. & Wibowo-Teale, A. M. Real-time time-dependent self-consistent field methods with dynamic magnetic fields. *The Journal of Chemical Physics* **159**, 104102 (2023), [DOI](https://doi.org/10.1063/5.0160317).

## Development history

### The inception

QSym² began in late 2017 as a subpackage named `SymmetryTools` in QCMagic, a set of code written in Python 2 used in the Thom group at the University of Cambridge, UK for manipulating multiple self-consistent-field (SCF) solutions calculated in Q-Chem.
`SymmetryTools` consisted largely of:

- a Python reimplementation of the [Beruski&ndash;Vidal algorithm](http://doi.wiley.com/10.1002/jcc.23493) for locating symmetry elements in molecules,
- an implementation of rotation matrices for real spherical harmonics formulated by [Ivanic and Ruedenberg](https://pubs.acs.org/doi/10.1021/jp953350u), and
- an implementation of generalised transformation matrices between Cartesian and pure spherical harmonic Gaussians based on the formulation by [Schlegel and Frisch](http://doi.wiley.com/10.1002/qua.560540202).

At the time, `SymmetryTools` was developed by the author (BCH) as part of his doctoral research under the advice of Dr Alex J. W. Thom to categorise symmetry-broken SCF solutions in octahedral transition-metal complexes by their symmetry.
This was achieved by a method that would eventually evolve into the generic [orbit-based representation analysis](../methodologies/orbit-based-representation-analysis.md) method of QSym².
However, there was no abstract group structure, nor was there any capability to generate character tables.

### The Python 3 re-development

In recognition of the deprecation of Python 2 on 1^st^ January 2020, the author decided that `SymmetryTools` must be re-developed in a way that embraces modern DevOps practice and that strives for generality so that it can be used by many quantum-chemistry tools.
And so, over the period of the first COVID-19 national lockdown that commenced in March 2020, A Python 3 package named [`poly-inspect`](https://gitlab.com/bangconghuynh/poly-inspect) came into existence.

The primary goal of `poly-inspect` at the time was to analyse and visualise multiple SCF solutions as vertices of high-dimensional polytopes, hence the name.
However, `poly-inspect` also reimplemented all features of `SymmetryTools` with proper documentation and unit testing.
New features were also added, most notably the ability to manually construct unitary-represented character tables for grey groups ($\mathcal{G} + \hat{\theta} \mathcal{G}$) and groups containing the complex-conjugation operation ($\mathcal{G} + \hat{K} \mathcal{G}$).

Although at this stage, `poly-inspect` still lacked the ability to generate character tables, an API for a generic quantum-chemical symmetry analysis library began to emerge.

### The integration with QUEST

In February 2022, during a post-doctoral research project undertaken in the Teale group at the University of Nottingham, UK, the author and Prof. Andrew M. Teale realised that a symmetry consideration was needed to move forward.
This prompted the integration of `poly-inspect` with [QUEST](https://quest.codes/) and presented the first real-life test for the API of `poly-inspect`.
Fortunately, minimal effort was required to port over all symmetry analysis functionalities of `poly-inspect` to QUEST.

However, `poly-inspect`'s inability for character table generation meant that users of QUEST who would like to obtain symmetry analysis results for their calculations would have had to carry out many manual steps to obtain definitive symmetry assignments.
To remedy this, 
