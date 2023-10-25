The following instructions for installing QSym² from source are specific to Linux operating systems. On Microsoft Windows platforms, Windows Subsystem for Linux is recommended.

## Binary compilation
The following instructions assume that the `full` feature is to be installed on a Debian/Ubuntu distro to make available the `qsym2` binary.

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
    or by downloading a compressed tarball release and extracting it.

5. Inside the source code directory, install QSym² by running the following command:
    ```bash
    cargo install --features full --path .
    ```
    which will compile and install the `qsym2` binary into `$HOME/.cargo/bin` to allow for system-wide availability.

    Note that the `--features` option in the command above specifies that the `full` feature is to be installed. This option accepts a comma-separated list of features listed in the [**Rust features**](prerequisites.md/#rust-features) section and can be modified to select other features to be installed as appropriate.
    
    It is possible to install the `qsym2` binary into a different path by running
    ```bash
    cargo install --features full --path . --root custom/install/path/for/qsym2
    ```
    instead. The custom path `/custom/install/path/for/qsym2` must then be added to the `$PATH` environment variable to make `qsym2` discoverable by the operating system.

## Python-library compilation
The following instructions assume that the `openblas-static`, `integrals`, and `python` features are to be compiled on a Debian/Ubuntu distro and then installed as a Python library inside a conda environment. These features are specified in the [`pyproject.toml`](https://gitlab.com/bangconghuynh/qsym2/-/blob/master/pyproject.toml) file.

1. Follow steps 1 to 4 under the [**Binary compilation**](#binary-compilation) section above to install the required prerequisites.
2. Make sure that the Anaconda package manager is available on your system. Instructions for installing Anaconda on a Linux system can be found [here](https://docs.anaconda.com/free/anaconda/install/linux/).
3. Create a new conda environment named `qsym2-python` (or a different name of your choice) running Python 3.11 (or a different version of your choice):
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

