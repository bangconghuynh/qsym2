use itertools::Itertools;
use log4rs;
use nalgebra::Point3;
use ndarray::Array1;
use num_traits::ToPrimitive;

use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::CharacterTableDisplay;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;
use crate::sandbox::drivers::representation_analysis::pes::{
    PESRepAnalysisDriver, PESRepAnalysisParams,
};
use crate::sandbox::target::pes::PES;
use crate::symmetry::symmetry_group::UnitaryRepresentedSymmetryGroup;
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_drivers_pes_analysis_if7() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/if7.xyz");

    let afa_params = AngularFunctionRepAnalysisParams::default();

    let pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[1e-6])
        .distance_thresholds(&[1e-6])
        .field_origin_com(true)
        .time_reversal(true)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&pd_params)
        .xyz(Some(path.into()))
        .build()
        .unwrap();
    assert!(pd_driver.run().is_ok());
    let pd_res = pd_driver.result().unwrap();

    let grid_points = (-50..=50)
        .cartesian_product(-50..=50)
        .cartesian_product(-50..=50)
        .map(|((i, j), k)| {
            let x = 0.0 + i.to_f64().unwrap() * 0.02;
            let y = 0.0 + j.to_f64().unwrap() * 0.02;
            let z = 0.0 + k.to_f64().unwrap() * 0.02;
            Point3::new(x, y, z)
        })
        .collect_vec();

    let weight: Array1<f64> = Array1::from_iter(
        grid_points
            .iter()
            .map(|pt| (-(pt - Point3::<f64>::origin()).magnitude_squared()).exp()),
    );

    // ~~~~~~~~~~~
    // E1'' (xz^3)
    // ~~~~~~~~~~~
    let e1dd_pes = PES::<f64, _>::builder()
        .function(|pt| pt.x * pt.z.powi(3))
        .grid_points(grid_points.clone())
        .build()
        .unwrap();

    let pes_params = PESRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_overlap_eigenvalues(true)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut pes_driver = PESRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, f64, _>::builder()
        .parameters(&pes_params)
        .angular_function_parameters(&afa_params)
        .pes(&e1dd_pes)
        .weight(&weight)
        .symmetry_group(pd_res)
        .build()
        .unwrap();
    assert!(pes_driver.run().is_ok());
    assert_eq!(
        *pes_driver.result().unwrap().pes_symmetry.as_ref().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|^('')_(1)|").unwrap()
    );

    // ~~~~~~~~~~
    // E2'' (xyz)
    // ~~~~~~~~~~
    let e2dd_pes = PES::<f64, _>::builder()
        .function(|pt| pt.x * pt.y * pt.z)
        .grid_points(grid_points.clone())
        .build()
        .unwrap();

    let pes_params = PESRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_overlap_eigenvalues(true)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut pes_driver = PESRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, f64, _>::builder()
        .parameters(&pes_params)
        .angular_function_parameters(&afa_params)
        .pes(&e2dd_pes)
        .weight(&weight)
        .symmetry_group(pd_res)
        .build()
        .unwrap();
    assert!(pes_driver.run().is_ok());
    assert_eq!(
        *pes_driver.result().unwrap().pes_symmetry.as_ref().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|^('')_(2)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~
    // A2'' + E1' (z + yz^2)
    // ~~~~~~~~~~~~~~~~~~~~~
    let a2dde1d_pes = PES::<f64, _>::builder()
        .function(|pt| pt.z + 2.0 * pt.y * pt.z.powi(2))
        .grid_points(grid_points.clone())
        .build()
        .unwrap();

    let pes_params = PESRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_overlap_eigenvalues(true)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut pes_driver = PESRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, f64, _>::builder()
        .parameters(&pes_params)
        .angular_function_parameters(&afa_params)
        .pes(&a2dde1d_pes)
        .weight(&weight)
        .symmetry_group(pd_res)
        .build()
        .unwrap();
    assert!(pes_driver.run().is_ok());
    assert_eq!(
        *pes_driver.result().unwrap().pes_symmetry.as_ref().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|^('')_(2)| ⊕ ||E|^(')_(1)|").unwrap()
    );
}
