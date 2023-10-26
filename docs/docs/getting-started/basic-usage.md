# Basic usage

There are two main ways of running QSym²:

- via the command-line interface provided by the binary `qsym2`, or
- via the exposed Python bindings.

## Command-line interface

### Scope

This method is currently able to perform symmetry analysis of:

- Slater determinants,
- Hartree&ndash;Fock or Kohn&ndash;Sham molecular orbitals, and
- vibrational coordinates

that have been exported by Q-Chem 6 to a HDF5 file named `qarchive.h5` saved in the job's scratch directory,

or

- Slater determinants, and
- Hartree&ndash;Fock or Kohn&ndash;Sham molecular orbitals

that have been stored in binary files, together with other basis-set-related data.

### Instructions

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

Examples of symmetry analysis performed by QSym² for several Q-Chem calculations can be found in the project's [Wiki](https://gitlab.com/bangconghuynh/qsym2/-/wikis/home) page.


## Python interface

### Scope

This method is currently able to perform symmetry analysis of:

- Slater determinants,
- Hartree&ndash;Fock or Kohn&ndash;Sham molecular orbitals,
- vibrational coordinates, and
- one-electron densities

that can be computed directly in Python or read into Python from calculation files of quantum-chemistry packages, such as by the use of [cclib](https://cclib.github.io/). The main driver functions of QSym² are all exposed to Python, which means that they can be used and integrated into existing workflows flexibly.

### Instructions

To view the documentation for the Python API, execute the following Python commands inside the `qsym2-python` conda environment (see [Python-library compilation](installation.md/#python-library-compilation)), either interactively or in a Python script:
```python
import qsym2
help(qsym2)
```

An example Python script that performs symmetry analysis for self-consistent-field calculations from Orca output files (parsed by [cclib](https://cclib.github.io/)) can be found at [`utils/qsym2-orca.py`](https://gitlab.com/bangconghuynh/qsym2/-/blob/master/utils/qsym2-orca.py). This script requires the Python packages `mendeleev`, `cclib`, and `numpy` to run.

Another example where the Python bindings of QSym² are used extensively for the analysis of unitary and magnetic symmetry in the presence of external fields can be found in [QUEST](https://quest.codes/) and its complementary GUI, [QuestView](https://gitlab.com/Bspeake/questview).
