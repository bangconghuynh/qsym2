# QSym²

QSym² is a Rust program for **Q**uantum **Sym**bolic **Sym**metry analysis of quantum-chemical calculations.

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
- `integrals`: Enables the computation of $n$-centre overlap integrals and $n$-centre overlap integral derivatives within QSym²

#### Composite
- `standard`: Enables `openblas-static` and `qchem` features
- `full`: Enables `standard` and `integrals` features

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
| `integrals`        | <ul><li> `libssl-dev` (Debian/Ubuntu)</li> <li>`openssl-devel` (Rocky/Fedora/RHEL)</li></ul>      | Installs the TLS framework required for [`reqwest`](https://github.com/seanmonstar/reqwest)                                                                                                  |
| `python`           | Python, which is best managed via Anaconda                                                        | Installs the Python bindings for core functionalities of QSym²                                                                                                                               |

### Binary compilation
The following instructions assume that the `full` feature is to be installed on a Debian/Ubuntu distro to make available the `qsym2` binary.

1. Install the basic dependencies by running the following commands (sudo priveleges required):
    ```bash
    sudo apt-get update
    sudo apt-get install curl git
    ```

2. Install the Rust compiler by running the command below and following the on-screen instructions:
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```
    The default configurations should suffice in most cases. Ensure that the current shell is restarted or the command `source "$HOME/.cargo/env"` is run upon installation completion so that the Rust compiler and the `cargo` package manager can be found.

3. Install the feature-specific dependencies by running the following commands (sudo priveleges required):
    ```bash
    sudo apt-get install build-essential gfortran cmake libssl-dev
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
        cargo install --features full --path . --root /custom/install/path/for/qsym2
    ```
    instead. The custom path `/custom/install/path/for/qsym2` must then be added to the `$PATH` environment variable to make `qsym2` discoverable by the operating system.

### Python-library compilation
The following instructions assume that the `full` and `python` features are to be compiled on a Debian/Ubuntu distro and then installed as a Python library inside a conda environment.

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
    This command calls `pip` which automatically acquires the build tool `maturin` to compile and install QSym² as a Python library into the `qsym2-python` conda environment.

    The Python library `qsym2` can now be imported by any Python scripts running inside the `qsym2-python` conda environment.

## Usage


## Contributing


## License