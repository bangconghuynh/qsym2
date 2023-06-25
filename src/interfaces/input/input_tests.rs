use nalgebra::{Point3, Vector3};

use crate::drivers::representation_analysis::CharacterTableDisplay;
use crate::io::read_qsym2_yaml;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;

use super::Input;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_interfaces_input_serde_yaml() {
    let name = format!("{ROOT}/tests/input/test_input.yml");
    let inp = read_qsym2_yaml::<Input>(&name).unwrap();

    assert_eq!(
        inp.symmetry_group_detection.moi_thresholds,
        vec![1e-4, 1e-5, 1e-6]
    );
    assert_eq!(
        inp.symmetry_group_detection.distance_thresholds,
        vec![1e-5, 1e-6]
    );
    assert!(inp.symmetry_group_detection.time_reversal);
    assert_eq!(
        inp.symmetry_group_detection
            .fictitious_magnetic_fields
            .as_ref()
            .unwrap()[0]
            .0,
        Point3::new(0.0, 1.0, 0.0)
    );
    assert_eq!(
        inp.symmetry_group_detection
            .fictitious_magnetic_fields
            .as_ref()
            .unwrap()[0]
            .1,
        Vector3::new(1.0, 0.0, 0.0)
    );
    assert!(inp
        .symmetry_group_detection
        .fictitious_electric_fields
        .is_none());
    assert!(inp.symmetry_group_detection.field_origin_com);
    assert!(!inp.symmetry_group_detection.write_symmetry_elements);
    assert!(inp.symmetry_group_detection.result_save_name.is_none());

    assert_eq!(inp.representation_analysis.integrality_threshold, 1e-8);
    assert_eq!(
        inp.representation_analysis.linear_independence_threshold,
        1e-7
    );
    assert!(inp.representation_analysis.analyse_mo_symmetries);
    assert!(!inp.representation_analysis.use_magnetic_group);
    assert!(!inp.representation_analysis.use_double_group);
    assert!(matches!(
        inp.representation_analysis.symmetry_transformation_kind,
        SymmetryTransformationKind::Spatial
    ));
    assert!(matches!(
        inp.representation_analysis.write_character_table,
        Some(CharacterTableDisplay::Numerical)
    ));
    assert!(inp.representation_analysis.write_overlap_eigenvalues);
    assert_eq!(
        inp.representation_analysis.infinite_order_to_finite,
        Some(8)
    );
}
