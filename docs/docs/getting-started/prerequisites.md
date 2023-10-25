## Rust features

The QSym² crate defines the following features that can be specified at compilation via `cargo`. Some of these features are mutually exclusive.

### Linear algebra backend
There are six features defining six different ways a linear algebra backend can be configured for QSym². These are inherited from the [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) crate. One (and only one) of these must be specified:

- `openblas-static`: Downloads, builds OpenBLAS, and links statically
- `openblas-system`: Finds and links existing OpenBLAS in the system
- `netlib-static`: Downloads, builds LAPACK, and links statically
- `netlib-system`: Finds and links existing LAPACK in the system
- `intel-mkl-static`: Finds and links existing static Intel MKL in the system, or downloads and links statically if not found
- `intel-mkl-system`: Finds and links existing shared Intel MKL in the system

If the `*-static` backends give rise to numerical problems, please try installing the linear algebra backends directly (either via your system's package manager or by compiling from source) and then using the corresponding `*-system` backends.

### Interfaces
- `qchem`: Enables Q-Chem's HDF5 archive files to be read in and analysed
- `python`: Enables the Python bindings for several core functionalities

### Integrals
- `integrals`: Enables the computation of $n$-centre overlap integrals and $n$-centre overlap integral derivatives within QSym²

### Composite
- `standard`: Enables the `openblas-static` and `qchem` features
- `full`: Enables the `standard` and `integrals` features

## Dependencies

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
| `netlib-system`    | <ul><li>`liblapack-dev` (Debian/Ubuntu)</li> <li>`lapack-devel` (Rocky/Fedora/RHEL)</li></ul>     | Finds and links existing LAPACK in the system (see  [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))                                                         |
| `intel-mkl-static` | <ul><li> `pkg-config` (Debian/Ubuntu)</li> <li>`pkgconfig` (Rocky/Fedora/RHEL)</li></ul>          | Finds and links existing static Intel MKL in the system, or downloads and links statically if not found (see  [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))|
| `intel-mkl-system` | <ul><li> `pkg-config` (Debian/Ubuntu)</li> <li>`pkgconfig` (Rocky/Fedora/RHEL)</li></ul>          | Finds and links existing shared Intel MKL in the system (see  [ndarray-linalg documentation](https://github.com/rust-ndarray/ndarray-linalg))                                                |
| `qchem`            | `cmake`, `gcc`                                                                                    | Builds the HDF5 C library and links statically                                                                                                                                               |
| `integrals`        | <ul><li> `libssl-dev` (Debian/Ubuntu)</li> <li>`openssl-devel` (Rocky/Fedora/RHEL)</li></ul>      | Installs the TLS framework required for [reqwest](https://github.com/seanmonstar/reqwest)                                                                                                  |
| `python`           | Python, which is best managed via Anaconda                                                        | Installs the Python bindings for core functionalities of QSym²                                                                                                                               |
