use nalgebra::{Rotation3, Vector3};
use num_traits::ToPrimitive;

use super::*;
use crate::aux::atom::{Atom, AtomKind};
use crate::aux::molecule::Molecule;
use crate::aux::template_molecules;
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_chartab_symmetric_h8_twisted_alt_magnetic_field_bw_c4_c2() {
    let thresh = 1e-7;
    let angle = 0.2;
    let mut mol = template_molecules::gen_twisted_h8(angle);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            [
                Atom::new_special(
                    AtomKind::Magnetic(i % 2 == 0),
                    atom.coordinates
                        + Rotation3::new(
                            Vector3::z()
                                * (std::f64::consts::FRAC_PI_2 * (i % 4).to_f64().unwrap()
                                    + angle * (i.div_euclid(4)).to_f64().unwrap()),
                        ) * (0.3 * Vector3::x()),
                    thresh,
                )
                .expect("Unable to construct a special magnetic atom."),
                Atom::new_special(
                    AtomKind::Magnetic(i % 2 != 0),
                    atom.coordinates
                        - Rotation3::new(
                            Vector3::z()
                                * (std::f64::consts::FRAC_PI_2 * (i % 4).to_f64().unwrap()
                                    + angle * (i.div_euclid(4)).to_f64().unwrap()),
                        ) * (0.3 * Vector3::x()),
                    thresh,
                )
                .expect("Unable to construct a special magnetic atom."),
            ]
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);

    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    test_chartab_ordinary_group(&mol, thresh, "C2", &expected_irreps, None);

    let mag_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    test_chartab_magnetic_group(&mol, thresh, "C4", &mag_expected_irreps, None);
}

#[test]
fn test_chartab_symmetric_h8_alt_x_magnetic_field_bw_s4_c2() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, atom)| {
            let direction_id = i.div_euclid(2);
            if direction_id == 0 || direction_id == 3 {
                Some([
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 == 0),
                        atom.coordinates
                            + Rotation3::new(
                                Vector3::z()
                                    * (std::f64::consts::FRAC_PI_2
                                        * direction_id.to_f64().unwrap()),
                            ) * (0.1 * Vector3::x()),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 != 0),
                        atom.coordinates
                            - Rotation3::new(
                                Vector3::z()
                                    * (std::f64::consts::FRAC_PI_2
                                        * direction_id.to_f64().unwrap()),
                            ) * (0.1 * Vector3::x()),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                ])
            } else {
                None
            }
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);

    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    test_chartab_ordinary_group(&mol, thresh, "C2", &expected_irreps, None);

    let mag_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    test_chartab_magnetic_group(&mol, thresh, "S4", &mag_expected_irreps, None);
}

#[test]
fn test_chartab_symmetric_bf3_rad_magnetic_field_bw_d3h_d3() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .filter_map(|atom| {
            if atom.atomic_symbol == "F" {
                Some([
                    Atom::new_special(AtomKind::Magnetic(true), 1.1 * atom.coordinates, thresh)
                        .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(AtomKind::Magnetic(false), 0.9 * atom.coordinates, thresh)
                        .expect("Unable to construct a special magnetic atom."),
                ])
            } else {
                None
            }
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);

    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    test_chartab_ordinary_group(&mol, thresh, "D3", &expected_irreps, None);

    let mag_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^('')|").unwrap(),
    ];
    test_chartab_magnetic_group(&mol, thresh, "D3h", &mag_expected_irreps, None);
}

#[test]
fn test_chartab_symmetric_tan_rad_magnetic_field_bw_c3h_c3() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, atom)| {
            if atom.atomic_symbol == "F" {
                Some([
                    Atom::new_special(
                        AtomKind::Magnetic(true),
                        atom.coordinates
                            + Rotation3::new(
                                Vector3::z()
                                    * (2.0
                                        * std::f64::consts::FRAC_PI_3
                                        * (i % 3).to_f64().unwrap()),
                            ) * (0.1 * Vector3::new(0.0, 1.0, 0.0)),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(
                        AtomKind::Magnetic(false),
                        atom.coordinates
                            - Rotation3::new(
                                Vector3::z()
                                    * (2.0
                                        * std::f64::consts::FRAC_PI_3
                                        * (i % 3).to_f64().unwrap()),
                            ) * (0.1 * Vector3::new(0.0, 1.0, 0.0)),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                ])
            } else {
                None
            }
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);

    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    test_chartab_ordinary_group(&mol, thresh, "C3", &expected_irreps, None);

    let mag_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|^(')|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|^('')|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|^('')|").unwrap(),
    ];
    test_chartab_magnetic_group(&mol, thresh, "C3h", &mag_expected_irreps, None);
}

#[test]
fn test_chartab_symmetric_bf3_alt_magnetic_field_bw_c2v_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, atom)| {
            if atom.atomic_symbol == "F" {
                Some([
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 == 0),
                        atom.coordinates + Vector3::new(0.0, 0.0, 1.0),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 != 0),
                        atom.coordinates + Vector3::new(0.0, 0.0, -1.0),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                ])
            } else {
                None
            }
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);
    verify_bw_c2v_cs(&mol, thresh);
}

#[test]
fn test_chartab_symmetric_xef4_rad_magnetic_field_bw_c4v_c2v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, atom)| {
            if atom.atomic_symbol == "F" {
                Some([
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 == 0),
                        1.1 * atom.coordinates + 0.1 * Vector3::z(),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 != 0),
                        0.9 * atom.coordinates - 0.1 * Vector3::z(),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                ])
            } else {
                None
            }
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);

    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
    ];
    test_chartab_ordinary_group(&mol, thresh, "C2v", &expected_irreps, None);

    let mag_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    test_chartab_magnetic_group(&mol, thresh, "C4v", &mag_expected_irreps, None);
}

#[test]
fn test_chartab_symmetric_xef4_alt_z_magnetic_field_bw_d4h_d2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, atom)| {
            if atom.atomic_symbol == "F" {
                Some([
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 == 0),
                        atom.coordinates + Vector3::new(0.0, 0.0, 1.0),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 != 0),
                        atom.coordinates + Vector3::new(0.0, 0.0, -1.0),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                ])
            } else {
                None
            }
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);

    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3u)|").unwrap(),
    ];
    test_chartab_ordinary_group(&mol, thresh, "D2h", &expected_irreps, None);

    let mag_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
    ];
    test_chartab_magnetic_group(&mol, thresh, "D4h", &mag_expected_irreps, None);
}

#[test]
fn test_chartab_symmetric_xef4_alt_xy_magnetic_field_bw_d4h_d2d() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, atom)| {
            if atom.atomic_symbol == "F" {
                Some([
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 == 0),
                        atom.coordinates
                            + Rotation3::new(
                                Vector3::z()
                                    * (std::f64::consts::FRAC_PI_2
                                        * ((i - 1) % 4).to_f64().unwrap()),
                            ) * (0.1 * Vector3::new(1.0, 1.0, 0.0)),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 != 0),
                        atom.coordinates
                            - Rotation3::new(
                                Vector3::z()
                                    * (std::f64::consts::FRAC_PI_2
                                        * ((i - 1) % 4).to_f64().unwrap()),
                            ) * (0.1 * Vector3::new(1.0, 1.0, 0.0)),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                ])
            } else {
                None
            }
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);

    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    test_chartab_ordinary_group(&mol, thresh, "D2d", &expected_irreps, None);

    let mag_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
    ];
    test_chartab_magnetic_group(&mol, thresh, "D4h", &mag_expected_irreps, None);
}

#[test]
fn test_chartab_symmetric_xef4_alt_x_magnetic_field_bw_c4h_s4() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, atom)| {
            if atom.atomic_symbol == "F" {
                Some([
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 == 0),
                        atom.coordinates
                            + Rotation3::new(
                                Vector3::z()
                                    * (std::f64::consts::FRAC_PI_2
                                        * ((i - 1) % 4).to_f64().unwrap()),
                            ) * (0.1 * Vector3::x()),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 != 0),
                        atom.coordinates
                            - Rotation3::new(
                                Vector3::z()
                                    * (std::f64::consts::FRAC_PI_2
                                        * ((i - 1) % 4).to_f64().unwrap()),
                            ) * (0.1 * Vector3::x()),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                ])
            } else {
                None
            }
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);

    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    test_chartab_ordinary_group(&mol, thresh, "S4", &expected_irreps, None);

    let mag_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
    ];
    test_chartab_magnetic_group(&mol, thresh, "C4h", &mag_expected_irreps, None);
}

#[test]
fn test_chartab_symmetric_h8_alt_x_magnetic_field_bw_c4h_c2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            let direction_id = i.div_euclid(2);
            [
                Atom::new_special(
                    AtomKind::Magnetic(i % 2 == 0),
                    atom.coordinates
                        + Rotation3::new(
                            Vector3::z()
                                * (std::f64::consts::FRAC_PI_2 * direction_id.to_f64().unwrap()),
                        ) * (0.1 * Vector3::x()),
                    thresh,
                )
                .expect("Unable to construct a special magnetic atom."),
                Atom::new_special(
                    AtomKind::Magnetic(i % 2 != 0),
                    atom.coordinates
                        - Rotation3::new(
                            Vector3::z()
                                * (std::f64::consts::FRAC_PI_2 * direction_id.to_f64().unwrap()),
                        ) * (0.1 * Vector3::x()),
                    thresh,
                )
                .expect("Unable to construct a special magnetic atom."),
            ]
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);

    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
    ];
    test_chartab_ordinary_group(&mol, thresh, "C2h", &expected_irreps, None);

    let mag_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
    ];
    test_chartab_magnetic_group(&mol, thresh, "C4h", &mag_expected_irreps, None);
}
