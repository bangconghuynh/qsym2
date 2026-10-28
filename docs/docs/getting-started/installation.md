---
title: Installation
description: Installation instructions for QSym²
---

# Installation

The following instructions for installing QSym² from source are specific to Linux operating systems. On Microsoft Windows platforms, Windows Subsystem for Linux is recommended.

There are two main ways of installing QSym²:

- as a Python library named `qsym2` providing the exposed Python bindings, or
- as a binary named `qsym2` providing the command-line interface.

Both methods can be configured via [Rust features](prerequisites.md/#rust-features).


## Python-library compilation

The following instructions assume that the `openblas-static`, `integrals`, and `python` features are to be compiled on a Linux operating system and then installed as a Python library inside a conda environment. These features are specified in the [`pyproject.toml`](https://gitlab.com/bangconghuynh/qsym2/-/blob/master/pyproject.toml) file.

1. Make sure that the Anaconda/Miniconda package manager is available on your system. Instructions for installing Anaconda/Miniconda on a Linux system can be found [here](https://docs.anaconda.com/free/anaconda/install/linux/).

2. Create a new conda environment named `qsym2-python` (or a different name of your choice) running Python 3.14 (or a different version of your choice):
    ```bash
    conda create -n qsym2-python -c conda-forge \
        python=3.14 pip pkg-config openssl make cmake \
        c-compiler cxx-compiler fortran-compiler rust \
        binutils
    ```
    and then activate this environment:
    ```bash
    conda activate qsym2-python
    ```

3. (Optional) If your system does not have system CA certificates, execute the following command to use SSL certificates that come with the conda environment:
    ```bash
    export SSL_CERT_FILE="$CONDA_PREFIX/ssl/cacert.pem"
    ```

4. Obtain the latest compressed tarball release of QSym² from [here](https://gitlab.com/bangconghuynh/qsym2/-/releases), either by downloading the `tar.gz` file through a browser or by using `wget`:
    ```bash
    wget https://gitlab.com/bangconghuynh/qsym2/-/archive/vX.Y.Z/qsym2-vX.Y.Z.tar.gz
    ```
    where `vX.Y.Z` is the desired version.

5. Execute
    ```bash
    pip install qsym2-vX.Y.Z.tar.gz
    ```
    where `vX.Y.Z` is the desired version.

    It is also possible to bypass step 4 and run `pip install` directly on the `tar.gz` link:
    ```bash
    pip install https://gitlab.com/bangconghuynh/qsym2/-/archive/vX.Y.Z/qsym2-vX.Y.Z.tar.gz
    ```

    This command automatically acquires the build tool `maturin` to compile and install QSym² as a Python library into the `qsym2-python` conda environment. This library has the `openblas-static`, `integrals`, and `python` features enabled due to the specification in the `pyproject.toml` file.

    The Python library `qsym2` can now be imported by any Python scripts running inside the `qsym2-python` conda environment.


## Binary compilation

The following instructions assume that the `full` feature is to be installed on a Linux operating system to make available the `qsym2` binary.

1. Install the basic dependencies by running the following commands (sudo privileges required):

    === "Debian/Ubuntu"
        ```bash
        sudo apt-get update
        sudo apt-get install curl git libssl-dev pkg-config
        ```

    === "Rocky/Fedora/RHEL"
        ```bash
        sudo dnf update
        sudo dnf install curl git-all openssl-devel pkgconfig
        ```

2. Install the Rust compiler by running the command below and following the on-screen instructions:
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```
    The default configurations should suffice in most cases. Ensure that the current shell is restarted or the command `source "$HOME/.cargo/env"` is run upon installation completion so that the Rust compiler and the `cargo` package manager can be found.

3. Install the feature-specific dependencies by running the following command (sudo privileges required):

    === "Debian/Ubuntu"
        ```bash
        sudo apt-get install build-essential gfortran cmake
        ```

    === "Rocky/Fedora/RHEL"
        ```bash
        sudo dnf install make automake gcc gcc-c++ kernel-devel gfortran cmake
        ```

4. Obtain the source code of QSym² either via `git`:
    ```bash
    git clone https://gitlab.com/bangconghuynh/qsym2.git
    ```
    or by downloading a compressed tarball release from [here](https://gitlab.com/bangconghuynh/qsym2/-/releases) and extracting it:
    ```bash
    wget https://gitlab.com/bangconghuynh/qsym2/-/archive/vX.Y.Z/qsym2-vX.Y.Z.tar.gz
    tar -xvf qsym2-vX.Y.Z.tar.gz
    cd qsym2-vX.Y.Z
    ```
    where `vX.Y.Z` is the desired version.

5. Inside the source code directory, install QSym² by running the following command:
    ```bash
    cargo install --features full --path .
    ```
    which will compile and install the `qsym2` binary into `$HOME/.cargo/bin` to allow for system-wide availability.

    Note that the `--features` option in the command above specifies that the `full` feature is to be installed. This option accepts a comma-separated list of features listed in the [**Rust features**](prerequisites.md/#rust-features) section and can be modified to select other features to be installed as appropriate.
    
    It is possible to install the `qsym2` binary into a different path by running
    ```bash
    cargo install --features full --path . --root /custom/install/path/for/qsym2
    ```
    instead. The custom binary path `/custom/install/path/for/qsym2/bin` must then be added to the `$PATH` environment variable to make the `qsym2` binary discoverable by the operating system.
