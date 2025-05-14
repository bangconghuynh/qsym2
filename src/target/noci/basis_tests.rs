// use env_logger;
use anyhow::format_err;
use ndarray::array;

use num_complex::Complex64;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::geometry::Transform;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{BasisAngularOrder, BasisAtom, BasisShell, CartOrder, ShellOrder};
use crate::group::{GroupProperties, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::{Basis, OrbitBasis};

#[test]
fn test_orbit_basis_transformation_h2o_cs() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_o0 = Atom::from_xyz("O +0.0000000 +0.0000000 +0.0000000", &emap, 1e-6).unwrap();
    let atm_h0 = Atom::from_xyz("H +1.0000000 +1.0000000 +0.0000000", &emap, 1e-6).unwrap();
    let atm_h1 = Atom::from_xyz("H -1.0000000 +1.1000000 +0.0000000", &emap, 1e-6).unwrap();

    let bsc_s = BasisShell::new(0, ShellOrder::Cart(CartOrder::qchem(0)));

    let batm_o0 = BasisAtom::new(&atm_o0, &[bsc_s.clone()]);
    let batm_h0 = BasisAtom::new(&atm_h0, &[bsc_s.clone()]);
    let batm_h1 = BasisAtom::new(&atm_h1, &[bsc_s]);

    let bao_h2o = BasisAngularOrder::new(&[batm_o0, batm_h0, batm_h1]);
    let mol_h2o =
        Molecule::from_atoms(&[atm_o0.clone(), atm_h0.clone(), atm_h1.clone()], 1e-7).recentre();

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol_h2o)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_cs = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true).unwrap();
    let group_u_cs_grey = UnitaryRepresentedGroup::from_molecular_symmetry(&magsym, None).unwrap();

    #[rustfmt::skip]
    let calpha = array![
        [ Complex64::new(1.0, 0.5) ],
        [ Complex64::new(0.2, 0.3) ],
        [ Complex64::new(0.6, 0.2) ],
    ];
    #[rustfmt::skip]
    let cbeta = array![
        [ Complex64::new(0.0, 1.0) ],
        [ Complex64::new(0.0, 0.0) ],
        [ Complex64::new(0.5, 0.2) ],
    ];
    let oalpha = array![1.0];
    let obeta = array![1.0];
    let det = SlaterDeterminant::<Complex64, SpinConstraint>::builder()
        .coefficients(&[calpha.clone(), cbeta.clone()])
        .occupations(&[oalpha, obeta])
        .bao(&bao_h2o)
        .mol(&mol_h2o)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-7)
        .build()
        .unwrap();

    // -----------
    // Orbit basis
    // -----------

    // Cs
    let orbit_basis_u_cs = OrbitBasis::builder()
        .group(&group_u_cs)
        .origins(vec![det.clone()])
        .action(|g, det| det.sym_transform_spatial(g).map_err(|err| format_err!(err)))
        .build()
        .unwrap();
    let orbit_basis_u_cs_elements = orbit_basis_u_cs
        .iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let sigma_orbit_basis_u_cs = orbit_basis_u_cs
        .sym_transform_spatial(group_u_cs.get_index(1).as_ref().unwrap())
        .unwrap();
    let sigma_orbit_basis_u_cs_elements = sigma_orbit_basis_u_cs
        .iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(orbit_basis_u_cs_elements, sigma_orbit_basis_u_cs_elements);

    // Cs'
    let orbit_basis_u_cs_grey = OrbitBasis::builder()
        .group(&group_u_cs_grey)
        .origins(vec![det.clone()])
        .action(|g, det| {
            det.sym_transform_spatial_with_spintimerev(g)
                .map_err(|err| format_err!(err))
        })
        .build()
        .unwrap();
    let orbit_basis_u_cs_grey_elements = orbit_basis_u_cs_grey
        .iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let theta_orbit_basis_u_cs_grey = orbit_basis_u_cs_grey
        .sym_transform_spatial_with_spintimerev(group_u_cs_grey.get_index(2).as_ref().unwrap())
        .unwrap();
    let theta_orbit_basis_u_cs_grey_elements = theta_orbit_basis_u_cs_grey
        .iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        orbit_basis_u_cs_grey_elements[2],
        theta_orbit_basis_u_cs_grey_elements[0]
    );
    assert_eq!(
        orbit_basis_u_cs_grey_elements[3],
        theta_orbit_basis_u_cs_grey_elements[1]
    );
    assert_eq!(
        orbit_basis_u_cs_grey_elements[0].coefficients()[0],
        -theta_orbit_basis_u_cs_grey_elements[2].coefficients()[0].clone()
    );
    assert_eq!(
        orbit_basis_u_cs_grey_elements[0].coefficients()[1],
        -theta_orbit_basis_u_cs_grey_elements[2].coefficients()[1].clone()
    );
    assert_eq!(
        orbit_basis_u_cs_grey_elements[1].coefficients()[0],
        -theta_orbit_basis_u_cs_grey_elements[3].coefficients()[0].clone()
    );
    assert_eq!(
        orbit_basis_u_cs_grey_elements[1].coefficients()[1],
        -theta_orbit_basis_u_cs_grey_elements[3].coefficients()[1].clone()
    );

    let k_theta_orbit_basis_u_cs_grey = orbit_basis_u_cs_grey
        .sym_transform_spatial_with_spintimerev(group_u_cs_grey.get_index(2).as_ref().unwrap())
        .and_then(|orbit| {
            orbit.sym_transform_spatial(group_u_cs_grey.get_index(2).as_ref().unwrap())
        })
        .unwrap();
    let k_theta_orbit_basis_u_cs_grey_elements = k_theta_orbit_basis_u_cs_grey
        .iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        k_theta_orbit_basis_u_cs_grey_elements[0].coefficients()[0],
        -cbeta.clone(),
    );
    assert_eq!(
        k_theta_orbit_basis_u_cs_grey_elements[0].coefficients()[1],
        calpha.clone(),
    );
    assert_eq!(
        k_theta_orbit_basis_u_cs_grey_elements[2].coefficients()[0],
        -calpha.map(|v| v.conj()),
    );
    assert_eq!(
        k_theta_orbit_basis_u_cs_grey_elements[2].coefficients()[1],
        -cbeta.map(|v| v.conj()),
    );
}
