# QSym²

![QSym² Logo](/images/qsym2_logo_no_text.svg)

QSym² is a Rust program for **Q**uantum **Sym**bolic **Sym**metry analysis of quantum-chemical calculations.

[[_TOC_]]

## Installation from Source

The following instructions for installing QSym² from source are specific to Unix-like operating systems. On Microsoft Windows platforms, Windows Subsystem for Linux is recommended.

### Features

The QSym² crate defines the following features that can be specified at compilation via `cargo`. Some of these features are mutually exclusive.

#### Linear algebra backend
There are six features defining six different ways a linear algebra backend can be configured for QSym². These are inherited from the (ndarray-linalg)[https://github.com/rust-ndarray/ndarray-linalg] crate. One (and only one) of these must be specified:
- `openblas-static`: Downloads, builds OpenBLAS, and links statically
- `openblas-system`: Finds and links existing OpenBLAS in the system
- `netlib-static`: Downloads, builds LAPACK, and links statically
- `netlib-system`: Finds and links existing LAPACK in the system
- `intel-mkl-static`: Finds and links existing static Intel MKL in the system, or downloads and links statically if not found
- `intel-mkl-system`: Finds and links existing shared Intel MKL in the system

#### Interfaces
- `qchem`: Enables Q-Chem's HDF5 archive files to be read in and analysed
- `python`: Enables the Python bindings for several core functionalities

#### Integrals
- `integrals`: Enables the computation of $`n`$-centre overlap integrals and $`n`$-centre overlap integral derivatives within QSym²

#### Composite
- `standard`: Enables the `openblas-static` and `qchem` features
- `full`: Enables the `standard` and `integrals` features

### Dependencies

The installation of QSym² requires the following:
- Common:
  * `curl` for installing the Rust compiler
  * the Rust compiler and the `cargo` package manager
  * `git` for obtaining the source code of QSym²
  * `libssl-dev` (Debian/Ubuntu) or `openssl-devel` (Rocky/Fedora/RHEL)
  * `pkg-config` (Debian/Ubuntu) or `pkgconfig` (Rocky/Fedora/RHEL)

- Feature-specific:

| Feature            | Dependencies                                                                                      | Notes                                                                                                                                                                                        |
|--------------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `openblas-static`  | `make`, `gcc`, `gfortran`                                                                         | Builds OpenBLAS and links statically (see [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))                                                                    |
| `openblas-system`  | <ul><li>`libopenblas-dev` (Debian/Ubuntu)</li> <li>`openblas-devel` (Rocky/Fedora/RHEL)</li></ul> | Finds and links existing OpenBLAS in the system (see [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))                                                         |
| `netlib-static`    | `make`, `gfortran`                                                                                | Builds LAPACK and links statically (see [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))                                                                      |
| `netlib-system`    | <ul><li>`liblapack-dev` (Debian/Ubuntu)</li> <li>`lapack-devel` (Rocky/Fedora/RHEL)</li></ul>     | Finds and links existing LAPACK in the system (see  [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg) )                                                         |
| `intel-mkl-static` | <ul><li> `pkg-config` (Debian/Ubuntu)</li> <li>`pkgconfig` (Rocky/Fedora/RHEL)</li></ul>          | Finds and links existing static Intel MKL in the system, or downloads and links statically if not found (see  [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg) |
| `intel-mkl-system` | <ul><li> `pkg-config` (Debian/Ubuntu)</li> <li>`pkgconfig` (Rocky/Fedora/RHEL)</li></ul>          | Finds and links existing shared Intel MKL in the system (see  [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg)                                                 |
| `qchem`            | `cmake`, `gcc`                                                                                    | Builds the HDF5 C library and links statically                                                                                                                                               |
| `integrals`        | <ul><li> `libssl-dev` (Debian/Ubuntu)</li> <li>`openssl-devel` (Rocky/Fedora/RHEL)</li></ul>      | Installs the TLS framework required for [reqwest](https://github.com/seanmonstar/reqwest)                                                                                                  |
| `python`           | Python, which is best managed via Anaconda                                                        | Installs the Python bindings for core functionalities of QSym²                                                                                                                               |

### Binary compilation
The following instructions assume that the `full` feature is to be installed on a Debian/Ubuntu distro to make available the `qsym2` binary.

1. Install the basic dependencies by running the following commands (sudo priveleges required):
    ```bash
    sudo apt-get update
    sudo apt-get install curl git libssl-dev pkg-config
    ```

2. Install the Rust compiler by running the command below and following the on-screen instructions:
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```
    The default configurations should suffice in most cases. Ensure that the current shell is restarted or the command `source "$HOME/.cargo/env"` is run upon installation completion so that the Rust compiler and the `cargo` package manager can be found.

3. Install the feature-specific dependencies by running the following commands (sudo priveleges required):
    ```bash
    sudo apt-get install build-essential gfortran cmake
    ```
4. Obtain the source code of QSym² either via `git`:
    ```bash
    git clone https://gitlab.com/bangconghuynh/qsym2.git
    ```
    or by downloading a release compressed tarball and extracting it.

5. Inside the source code directory, install QSym² by running the following command:
    ```bash
    cargo install --features full --path .
    ```
    which will compile and install the `qsym2` binary into `$HOME/.cargo/bin` to allow for system-wide availability.
    
    It is possible to install the `qsym2` binary into a different path by running
    ```bash
        cargo install --features full --path . --root custom/install/path/for/qsym2
    ```
    instead. The custom path `/custom/install/path/for/qsym2` must then be added to the `$PATH` environment variable to make `qsym2` discoverable by the operating system.

### Python-library compilation
The following instructions assume that the `openblas-static`, `integrals`, and `python` features are to be compiled on a Debian/Ubuntu distro and then installed as a Python library inside a conda environment. These features are specified in the `pyproject.toml` file.

1. Follow steps 1 to 4 under the *Binary compilation* section above to install the required prerequisites.
2. Make sure that the Anaconda package manager is available on your system. Instructions for installing Anaconda on a Linux system can be found [here](https://docs.anaconda.com/free/anaconda/install/linux/).
3. Create a new conda environment named `qsym2-python` (or a different name of your choice) running Python 3.11 (or a different version your your choice):
    ```bash
    conda create -n qsym2-python python=3.11
    ```
    and then activate this environment:
    ```bash
    conda activate qsym2-python
    ```
4. From inside the source code directory, execute
    ```bash
    pip install .
    ```
    This command calls `pip` which automatically acquires the build tool `maturin` to compile and install QSym² as a Python library into the `qsym2-python` conda environment. This library has the `openblas-static`, `integrals`, and `python` features enabled due to the specification in the `pyproject.toml` file.

    The Python library `qsym2` can now be imported by any Python scripts running inside the `qsym2-python` conda environment.

## Usage

There are two main ways of running QSym²: either via the command-line interface provided by the binary `qsym2`, or via the exposed Python bindings.

### Command-line interface

This method is currently able to perform symmetry analysis of:
- Slater determinants,
- Hartree--Fock or Kohn--Sham molecular orbitals, and
- vibrational coordinates

that have been exported by Q-Chem 6 in a HDF5 file named `qarchive.h5` saved in the job's scratch directory,

or
- Slater determinants, and
- Hartree--Fock or Kohn--Sham molecular orbitals

that have been stored in binary files, alongside the overlap matrix of the uderlying atomic-orbital basis functions that has also been stored in a binary format.

The command-line interface supports several subcommands and options:
```bash
qsym2 -h
```

```
A program for Quantum Symbolic Symmetry

Usage: qsym2 <COMMAND>

Commands:
  template
          Generates a template YAML configuration file and exits
  run
          Runs an analysis calculation and exits
  help
          Print this message or the help of the given subcommand(s)

Options:
  -h, --help
          Print help
  -V, --version
          Print version
```

The subcommand `template`, runnable as
```bash
qsym2 template
```
generates a template configuration YAML file populated with some default control parameters. This file can be modified to adjust the parameters to suit the calculation at hand.

The subcommand `run`, runnable as
```bash
qsym2 run -c path/to/config -o output_name
```
takes a configuration YAML file as a parameter, performs the specified symmetry analysis, and displays the results in the specified output file.

Examples of symmetry analysis performed by QSym² for several Q-Chem calculations can be found in the *Tutorials* section of the project's Wiki page.


### Python interface

This method is currently able to perform symmetry analysis of:
- Slater determinants,
- Hartree--Fock or Kohn--Sham molecular orbitals,
- vibrational coordinates, and
- one-electron densities

that can be computed directly in Python or read into Python from calculation files of quantum-chemistry packages, such as by the use of [cclib](https://cclib.github.io/). The main driver functions of QSym² are all exposed to Python, which means that they can be used and integrated into existing workflows flexibly. 

An example Python script that performs symmetry analysis for self-consistent-field calculations from Orca output files (parsed by [cclib](https://cclib.github.io/)) can be found at `utils/qsym2_orca.py`.


## Authors and acknowledgement

QSym² was developed by Dr Bang C. Huynh at the University of Nottingham, UK with scientific support from Prof. Andrew M. Wibowo-Teale and Dr Meilani Wibowo-Teale and financial support from the ERC grant under the *topDFT* project.

The logo for QSym², which is a stylised stellated octahedron, was designed with artistic support from Mr [Thinh Nguyen](https://www.linkedin.com/in/thinh-nguyen-a38b7856/).
