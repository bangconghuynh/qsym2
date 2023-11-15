---
title: Development
description: Development of QSym²
---

# Development

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

### The Python 3 redevelopment

In response to the deprecation of Python 2 on 1^st^ January 2020, the author decided that `SymmetryTools` must be redeveloped in a way that embraces modern DevOps practice and that strives for generality so that it can be used by many quantum-chemistry tools.
And so, over the period of the first COVID-19 national lockdown that commenced in March 2020, A Python 3 package named [`poly-inspect`](https://gitlab.com/bangconghuynh/poly-inspect) came into existence.

The primary goal of `poly-inspect` at the time was to analyse and visualise multiple SCF solutions as vertices of high-dimensional polytopes, hence the name.
However, `poly-inspect` also reimplemented all features of `SymmetryTools` with proper documentation and unit testing.
New features were also added, most notably the ability to represent character tables programmatically, although class structures, irreducible representations, and characters all had to be entered manually.

Even though `poly-inspect` still lacked the ability to generate character tables on-the-fly at this stage, an API for a generic quantum-chemical symmetry analysis library began to emerge.

### The integration with QUEST

In February 2022, during a post-doctoral research project undertaken in the Teale group at the University of Nottingham, UK, the author and Prof. Andrew M. Teale realised that a symmetry consideration was needed to move forward.
This prompted the integration of `poly-inspect` with [QUEST](https://quest.codes/), the electronic-structure calculation suite developed in the Teale group, and presented the first real-life test for the API of `poly-inspect`.
Fortunately, minimal effort was required to port over all symmetry analysis functionalities of `poly-inspect` to QUEST.

However, `poly-inspect`'s inability for character table generation meant that users of QUEST who would like to obtain symmetry analysis results for their calculations would have had to carry out many manual steps to obtain definitive symmetry assignments.
To remedy this, the author decided to explore options to implement an algorithm for generating character tables automatically based on the symmetry elements locatable by `poly-inspect`.
Work thus began to allow symmetry operations generatable from located geometrical symmetry elements to be accurately and efficiently represented computationally in such a way that also enables their multiplication to be performed with ease.

Once a multiplicative structure of symmetry operations had been implemented and tested, it became possible to transition from concrete symmetry-group structures to equivalent abstract group structures, which in turn enabled character tables to be constructed symbolically and efficiently over finite fields via the [Burnside&ndash;Dixon algorithm](https://doi.org/10.1007/BF02162877).
This was very exciting because, for the first time, it was possible to perform symmetry analysis for quantum-chemical calculations in any arbitrary point groups without needing pre-tabulated character tables or requiring that molecules have to be in some standard orientation.

As one of the most distintive features of QUEST is its ability to carry out non-perturbative electronic-structure calculations in the presence of external electric and magnetic fields, the added capability of symmetry analysis in any symmetry point groups means that it was then also possible to investigate the effects of external fields on the symmetry of quantum-chemical properties.
This in fact led to several pieces of joint work with Dr Tom J. P. Irons and Dr Meilani Wibowo in the group in which the understanding from symmetry helps us make sense of the behaviours of chemical systems under the influence of strong magnetic fields.

### The Rust redevelopment

Despite having a working symmetry analysis implementation in QUEST, the author was not entirely happy that it was written in Python, because of the known performance limitations that are inherent in this language.
Having been very impressed with the design of the modern system programming language Rust that prioritises safety and performance, the author decided that the symmetry analysis code would benefit from a redevelopment in Rust.
And so, around April 2022, the work to rebuild the entire symmetry analysis suite in Rust began.
This started out as a side project, but with the scientific support from Prof. Andrew M. Teale and Dr Meilani Wibowo, the redevelopment work gradually became a full-time project that constantly received ideas and suggestions for more functionalities.

During the initial redevelopment period, the Rust codebase was given the name `rusty-inspect`.
However, in the summer of 2022, after having used the abbreviation 'sym' on multiple occasions in the codebase to refer to both 'symmetry' and 'symbolic' or 'symbol', the author realised that the codebase could be renamed in such a way that would include both meanings of 'sym'.
It was a small leap then to arrive at the name QSym² that has gone on to become the official name for this program.

Most symmetry analysis functionalities that had been implemented in QUEST prior to this redevelopment were successfully ported to QSym² by early 2023.
However, it did not sit well with the author and his collaborators that QSym² was limited to only unitary symmetry representation analysis, given that the framework of QSym² and the type system of Rust are both sufficiently generic to go much further beyond this.
And thus commenced the period of feature expansions and code refactoring with the aim to distinguish implementations that are specific to a particular physical problem (*e.g.* point-transformation symmetry operations) from implementations that are fundamentally abstract because of the abstract structure of groups (*e.g.* character table generation).
This period occupied a good part of the first half of 2023.

The outcome from this period is a plethora of new features introduced to QSym² that, to the best of the author's knowledge, have not been seen before in any other quantum-chemical calculation programs.
These features include the abilities to handle magnetic symmetry via corepresentations, to incorporate spin symmetry explicitly via projective (co)representations, and to apply symmetry analysis to a wide range of linear-space quantities that go beyond Slater determinants such as electron densities and vibrational coordinates.
It was at this stage that the author and his collaborators felt that QSym² had reached a suitable standard for being released.
It was also during this period that the author had the pleasure to congratulate his collaborators on their holy matrimony.

Since then, the focus has been to test and improve QSym² rigorously to get it ready for a public release.
A major aspect of this preparation is the development of drivers, bindings, and interfaces to expose the features QSym² to end-users in multiple ways.
This has since enabled QSym² to work with QUEST, [Q-Chem](https://www.q-chem.com/), [Orca](https://orcaforum.kofo.mpg.de/), or even raw binary files.
These are by no means everything that QSym² will ever support, and interfaces with other quantum chemistry packages will continue to be added in the future.
But for now, these are enough to demonstrate that the design of QSym² is flexible and general enough to be used with a variety of programs.

## Acknowledgement

The author would like to acknowledge the following individuals for their support of QSym² in its development, however big or small:

- Prof. Andrew M. Wibowo-Teale (Nottingham, UK)
- Dr Meilani Wibowo-Teale (Nottingham, UK)
- Dr Tom J. P. Irons (Nottingham, UK)
- Dr Alex J. W. Thom (Cambridge, UK)
- Dr Benjamin Speake (Nottingham, UK)
- Prof. Paul Geerlings (Brussels, Belgium)
- Mr Benjamin Mokhtar (Paris, France)
- Dr Dr Aleksandra Foerster (Nottingham, UK)
- Mr Thinh Nguyen (Los Angeles, California, US)

The author would also like to acknowledge financial support from the following organisations for the entire duration of development of QSym² from its very beginning:

- European Research Council
- Cambridge Trust, Cambridge, UK
- Peterhouse, Cambridge, UK
