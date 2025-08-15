use std::path::Path;

use anyhow::{self, format_err};
use hdf5::{self, H5Type};
use itertools::Itertools;
use nalgebra::Point3;
use ndarray::{Array1, Array2, Array3, Array4, Axis, Ix3, Ix4};
use ndarray_linalg::Lapack;
use num_complex::ComplexFloat;
use periodic_table;

use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder, SpinorOrder,
};
use crate::target::determinant::SlaterDeterminant;

use crate::target::noci::backend::matelem::hamiltonian::HamiltonianAO;
use crate::target::noci::backend::matelem::overlap::OverlapAO;

/// Structure to store data from a `PySCF` SCF calculation.
pub(crate) struct PyScfData<T>
where
    T: ComplexFloat,
{
    /// The raw basis angular order of the basis. Each row contains the following:
    /// - atom index,
    /// - shell index
    /// - l
    /// - shell type: 0 (pure), 1 (Cartesian), or 2 (spinor)
    /// - even spatial part: 0 (false) or 1 (true)
    pub(crate) basis_bao: Array2<usize>,

    /// The one-electron integral matrix in the AO basis.
    pub(crate) integrals_onee_h: Array2<T>,

    /// The two-electron integral tensor in the AO basis.
    pub(crate) integrals_twoe_h: Array4<T>,

    /// The overlap matrix in the AO basis.
    pub(crate) integrals_sao: Array2<T>,

    /// The atoms in the molecule. Each element gives the atomic number of an atom.
    pub(crate) mol_atoms: Array1<usize>,

    /// The corresponding Cartesian coordinates of the atoms.
    pub(crate) mol_coords: Array2<f64>,

    /// The nuclear repulsion energy of the molecule.
    pub(crate) mol_enuc: T::Real,

    /// The SCF MO coefficients.
    pub(crate) scf_cs: Array3<T>,

    /// The occupation numbers of the MOs.
    pub(crate) scf_occs: Array2<T::Real>,

    /// The MO energies.
    pub(crate) scf_mo_energies: Array2<T>,

    /// The reference one-electron energy.
    pub(crate) scf_e_1: T,

    /// The reference two-electron energy.
    pub(crate) scf_e_2: T,

    /// The reference SCF energy.
    pub(crate) scf_e_scf: T,
}

impl<T> PyScfData<T>
where
    T: ComplexFloat,
{
    /// Obtains the molecule from the HDF5 data file.
    ///
    /// # Arguments
    ///
    /// * `emap` - A periodic table.
    /// * `thresh` - A threshold for atom and molecule comparison.
    pub(crate) fn get_mol(
        &self,
        emap: &ElementMap,
        thresh: f64,
    ) -> Result<Molecule, anyhow::Error> {
        let atoms = self
            .mol_atoms
            .iter()
            .zip(self.mol_coords.rows())
            .map(|(atm_i, atm_coords)| {
                let element = periodic_table::periodic_table()
                    .get(atm_i - 1)
                    .unwrap_or_else(|| {
                        panic!("Unable to obtain element with atomic number {atm_i}.")
                    })
                    .symbol;
                Atom::new_ordinary(
                    element,
                    Point3::new(atm_coords[0], atm_coords[1], atm_coords[2]),
                    emap,
                    thresh,
                )
            })
            .collect_vec();

        let mol = Molecule::from_atoms(&atoms, thresh);
        Ok(mol)
    }

    /// Obtains the basis angular order information from the HDF5 data file.
    ///
    /// # Arguments
    ///
    /// * `mol` - The associated molecule.
    pub(crate) fn get_bao<'a, 'b: 'a>(
        &'a self,
        mol: &'b Molecule,
    ) -> Result<BasisAngularOrder<'a>, anyhow::Error> {
        let basis_atoms = self
            .basis_bao
            .rows()
            .into_iter()
            .chunk_by(|row| row[0])
            .into_iter()
            .map(|(atom_index, rows)| {
                let atom = &mol.atoms[atom_index];
                let atom_shells = rows
                    .into_iter()
                    .map(|row| {
                        let l: u32 = row[2]
                            .try_into()
                            .expect("Unable to convert the angular momentum to `u32`.");
                        let st = if row[3] == 0 {
                            if l == 1 {
                                ShellOrder::Pure(
                                    PureOrder::new(&[1, -1, 0])
                                        .expect("Unable to construct a custom PureOrder."),
                                )
                            } else {
                                ShellOrder::Pure(PureOrder::increasingm(l))
                            }
                        } else if row[3] == 1 {
                            ShellOrder::Cart(CartOrder::lex(l))
                        } else if row[3] == 2 {
                            let spatial_even = row[4];
                            ShellOrder::Spinor(SpinorOrder::increasingm(l, spatial_even == 1, None))
                        } else {
                            panic!()
                        };
                        BasisShell::new(l, st)
                    })
                    .collect_vec();
                BasisAtom::new(atom, &atom_shells)
            })
            .collect_vec();
        Ok(BasisAngularOrder::new(&basis_atoms))
    }
}

impl<T> PyScfData<T>
where
    T: ComplexFloat + Lapack + TryFrom<<T as ComplexFloat>::Real>,
{
    /// Extracts the integrals from the HDF5 data file.
    pub(crate) fn get_integrals<SC: StructureConstraint + Clone>(
        &'_ self,
    ) -> Result<(OverlapAO<'_, T, SC>, HamiltonianAO<'_, T, SC>), anyhow::Error> {
        let overlap_ao = OverlapAO::<T, SC>::builder()
            .sao(self.integrals_sao.view())
            .build()?;
        let hamiltonian_ao = HamiltonianAO::<T, SC>::builder()
            .onee(self.integrals_onee_h.view())
            .twoe(self.integrals_twoe_h.view())
            .enuc(self.mol_enuc.try_into().map_err(|_| {
                format_err!("Unable to convert the nuclear repulsion energy into the correct type.")
            })?)
            .build()?;
        Ok((overlap_ao, hamiltonian_ao))
    }

    /// Extracts the Slater determinant from the HDF5 data file.
    ///
    /// # Arguments
    ///
    /// * `mol` - The associated molecule.
    /// * `bao` - The associated basis angular order information.
    /// * `sc` - The structure constraint.
    /// * `threshold` - A threshold for comparisons.
    pub(crate) fn get_slater_determinant<
        'a,
        'b: 'a,
        SC: StructureConstraint + Clone + std::fmt::Display,
    >(
        &'a self,
        mol: &'b Molecule,
        bao: &'b BasisAngularOrder,
        sc: SC,
        threshold: <T as ComplexFloat>::Real,
    ) -> Result<SlaterDeterminant<'a, T, SC>, anyhow::Error> {
        let cs = self
            .scf_cs
            .axis_iter(Axis(0))
            .into_iter()
            .map(|c| c.to_owned())
            .collect::<Vec<_>>();
        let occs = self
            .scf_occs
            .axis_iter(Axis(0))
            .into_iter()
            .map(|c| c.to_owned())
            .collect::<Vec<_>>();
        SlaterDeterminant::<T, SC>::builder()
            .coefficients(&cs)
            .occupations(&occs)
            .bao(&bao)
            .mol(&mol)
            .structure_constraint(sc)
            .complex_symmetric(false)
            .threshold(threshold)
            .build()
            .map_err(|err| format_err!(err))
    }
}

/// Reads a HDF5 data file and extracts the data into a `PyScfData` struct.
pub(crate) fn extract_pyscf_scf_data<P, T>(filename: P) -> Result<PyScfData<T>, anyhow::Error>
where
    P: AsRef<Path>,
    T: ComplexFloat + H5Type,
    T::Real: H5Type,
{
    let f = hdf5::File::open(&filename)?;

    let basis_bao = f.dataset("basis/bao")?.read_2d::<usize>()?;

    let integrals_onee_h = f.dataset("integrals/onee_h")?.read_2d::<T>()?;
    let integrals_twoe_h = f
        .dataset("integrals/twoe_h")?
        .read_dyn::<T>()?
        .into_dimensionality::<Ix4>()?;
    let integrals_sao = f.dataset("integrals/sao")?.read_2d::<T>()?;

    let mol_atoms = f.dataset("mol/atoms")?.read_1d::<usize>()?;
    let mol_coords = f.dataset("mol/coords")?.read_2d::<f64>()?;
    let mol_enuc = f.dataset("mol/enuc")?.read_scalar::<T::Real>()?;

    let scf_cs = f
        .dataset("scf/1/cs")?
        .read_dyn::<T>()?
        .into_dimensionality::<Ix3>()?;
    let scf_occs = f.dataset("scf/1/occs")?.read_2d::<T::Real>()?;
    let scf_mo_energies = f.dataset("scf/1/mo_energies")?.read_2d::<T>()?;
    let scf_e_1 = f.dataset("scf/1/e_1")?.read_scalar::<T>()?;
    let scf_e_2 = f.dataset("scf/1/e_2")?.read_scalar::<T>()?;
    let scf_e_scf = f.dataset("scf/1/e_scf")?.read_scalar::<T>()?;

    Ok(PyScfData {
        basis_bao,
        integrals_onee_h,
        integrals_twoe_h,
        integrals_sao,
        mol_atoms,
        mol_coords,
        mol_enuc,
        scf_cs,
        scf_occs,
        scf_mo_energies,
        scf_e_1,
        scf_e_2,
        scf_e_scf,
    })
}
