#![doc(
    html_logo_url = "https://qsym2.dev/assets/logos/qsym2_logo_no_text_no_background.svg",
    html_favicon_url = "https://qsym2.dev/assets/logos/qsym2_icon.png"
)]

//! # QSym²: A Program for Quantum Symbolic Symmetry
//!
//! QSym² is a program for **Q**uantum **Sym**bolic **Sym**metry analysis of
//! quantum-chemical calculations written in Rust with the following capabilities:
//! - on-the-fly generation of symbolic character tables,
//! - analysis of degeneracy and symmetry breaking effects in Abelian and non-Abelian groups,
//! - analysis of symmetry in external magnetic and electric fields,
//! - inclusion of antiunitary symmetry based on corepresentation theory, and
//! - inclusion of double-valued representations and corepresentations via explicit spin rotations
//!
//! for the following targets:
//! - Slater determinants,
//! - molecular orbitals,
//! - multi-determinantal wavefunctions obtained via non-orthogonal configuration interaction,
//! - real-space functions defined on a grid,
//! - electron densities, and
//! - vibrational coordinates.
//!
//! QSym² has been integrated with [QUEST](https://quest.codes/) and its complementary
//! GUI, [QuestView](https://gitlab.com/Bspeake/questview). QSym² can also work with
//! [Q-Chem](https://www.q-chem.com/) HDF5 archive files and
//! [Orca](https://orcaforum.kofo.mpg.de/index.php) output files.
//!
//! The main website for QSym² can be found [here](https://qsym2.dev).
//!
//! QSym² is hosted on [GitLab](https://gitlab.com/bangconghuynh/qsym2). Please submit
//! an issue there if you've encountered anything that is unclear or that you feel needs improving.
//!
//! This documentation details the public API of the `qsym2` crate.
//!
//! ## Help and support
//!
//! Developmental and programming issues with the crate `qsym2` can be reported on
//! [GitLab](https://gitlab.com/bangconghuynh/qsym2). For scientific enquiries such as how to run
//! the program QSym² for a particular quantum-chemical calculation or how to interpret
//! the results of QSym², please join the dedicated [Slack](https://qsym2.slack.com)
//! workspace using this [invite link](https://join.slack.com/t/qsym2/shared_invite/zt-24thj1y1x-JqvLtEA1xfQ9AylNRCDH9w).
//!
//! ## Getting started
//!
//! To use QSym² in your Rust project, simply add this crate to your project's
//! `Cargo.toml`. The available features defined by this crate are:
//!
//! ### Linear algebra backend
//!
//! There are six features defining six different ways a linear algebra backend can be configured
//! for QSym². These are inherited from the
//! [`ndarray-linalg`](https://docs.rs/ndarray-linalg/latest/ndarray_linalg/) crate. One
//! (and only one) of these must be enabled:
//! - `openblas-static`: Downloads, builds OpenBLAS, and links statically
//! - `openblas-system`: Finds and links existing OpenBLAS in the system
//! - `netlib-static`: Downloads, builds LAPACK, and links statically
//! - `netlib-system`: Finds and links existing LAPACK in the system
//! - `intel-mkl-static`: Finds and links existing static Intel MKL in the system, or downloads and
//!   links statically if not found
//! - `intel-mkl-system`: Finds and links existing shared Intel MKL in the system
//!
//! If the `*-static` backends give rise to numerical problems, please try installing the linear
//! algebra backends directly (either via your system's package manager or by compiling from source)
//! and then using the corresponding `*-system` backends.
//!
//! ### Interfaces
//! - `qchem`: Enables Q-Chem's HDF5 archive files to be read in and analysed
//! - `python`: Enables the Python bindings for several core functionalities
//!
//! ### Integrals
//! - `integrals`: Enables the computation of $`n`$-centre overlap integrals and $`n`$-centre
//!   overlap integral derivatives within QSym²
//!
//! ### Composite
//! - `standard`: Enables the `openblas-static` and `qchem` features
//! - `full`: Enables the `standard` and `integrals` features
//!
//! ### Developmental
//! - `sandbox`: Enables experimental features that are still being actively developed
//!
//! ## Dependencies
//!
//! The compilation of QSym² requires the following:
//! - Common:
//!   * `libssl-dev` (Debian/Ubuntu) or `openssl-devel` (Rocky/Fedora/RHEL)
//!   * `pkg-config` (Debian/Ubuntu) or `pkgconfig` (Rocky/Fedora/RHEL)
//!
//! - Feature-specific:
//!
//! | Feature            | Dependencies                                                                                      | Notes                                                                                                                                                                                        |
//! |--------------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
//! | `openblas-static`  | `make`, `gcc`, `gfortran`                                                                         | Builds OpenBLAS and links statically (see [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))                                                                    |
//! | `openblas-system`  | <ul><li>`libopenblas-dev` (Debian/Ubuntu)</li> <li>`openblas-devel` (Rocky/Fedora/RHEL)</li></ul> | Finds and links existing OpenBLAS in the system (see [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))                                                         |
//! | `netlib-static`    | `make`, `gfortran`                                                                                | Builds LAPACK and links statically (see [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))                                                                      |
//! | `netlib-system`    | <ul><li>`liblapack-dev` (Debian/Ubuntu)</li> <li>`lapack-devel` (Rocky/Fedora/RHEL)</li></ul>     | Finds and links existing LAPACK in the system (see  [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))                                                          |
//! | `intel-mkl-static` | <ul><li> `pkg-config` (Debian/Ubuntu)</li> <li>`pkgconfig` (Rocky/Fedora/RHEL)</li></ul>          | Finds and links existing static Intel MKL in the system, or downloads and links statically if not found (see  [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))|
//! | `intel-mkl-system` | <ul><li> `pkg-config` (Debian/Ubuntu)</li> <li>`pkgconfig` (Rocky/Fedora/RHEL)</li></ul>          | Finds and links existing shared Intel MKL in the system (see  [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))                                                |
//! | `qchem`            | `cmake`, `gcc`                                                                                    | Builds the HDF5 C library and links statically                                                                                                                                               |
//! | `integrals`        | <ul><li> `libssl-dev` (Debian/Ubuntu)</li> <li>`openssl-devel` (Rocky/Fedora/RHEL)</li></ul>      | Installs the TLS framework required for [reqwest](https://github.com/seanmonstar/reqwest)                                                                                                    |
//! | `python`           | Python, which is best managed via Anaconda                                                        | Installs the Python bindings for core functionalities of QSym²                                                                                                                               |
//!
//! ## Examples and usage
//!
//! For most items (structs, enums, functions, and traits), their usages are illustrated in test
//! functions. For more explanation, please consult this documentation.
//!
//! For usage of the compiled `qsym2` binary or Python library, please consult the `README.md` file
//! on [GitLab](https://gitlab.com/bangconghuynh/qsym2) and the documentations on
//! [QSym²'s website](https://qsym2.dev).
//!
//! ## License
//!
//! GNU Lesser General Public License v3.0.
//!
//! ## Authors and acknowledgement
//!
//! QSym² has been developed and maintained by
//! Dr [Bang C. Huynh](https://orcid.org/0000-0002-5226-4054) at the University of Nottingham, UK
//! since July 2022 with scientific support from Prof.
//! [Andrew M. Wibowo-Teale](https://orcid.org/0000-0001-9617-1143) and Dr
//! [Meilani Wibowo-Teale](https://orcid.org/0000-0003-2462-3328) and financial support from
//! the ERC grant under the *topDFT* project.
//!
//! The logo for QSym², which is a stylised stellated octahedron, was designed with
//! artistic support from Mr [Thinh Nguyen](https://www.linkedin.com/in/thinh-nguyen-a38b7856/).

macro_rules! count_exprs {
    () => (0);
    ($head:expr) => (1);
    ($head:expr, $($tail:expr),*) => (1 + count_exprs!($($tail),*));
}

macro_rules! replace_expr {
    ($_t:tt $sub:expr) => {
        $sub
    };
}

pub mod analysis;
pub mod angmom;
pub mod auxiliary;
pub mod basis;
pub mod bindings;
pub mod chartab;
pub mod drivers;
pub mod group;
#[cfg(feature = "integrals")]
pub mod integrals;
pub mod interfaces;
pub mod io;
pub mod permutation;
pub mod projection;
pub mod rotsym;
#[cfg(feature = "sandbox")]
pub mod sandbox;
pub mod symmetry;
pub mod target;
