use nalgebra::{Point3, Vector3};

use crate::drivers::representation_analysis::CharacterTableDisplay;
use crate::io::{read_qsym2_yaml, write_qsym2_yaml};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;

use super::{Input, SymmetryGroupDetectionInputKind};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_interfaces_input_symmetry_group_detection_parameters() {
    let name = format!("{ROOT}/tests/input/test_input_symmetry_group_detection_parameters.yml");
    let inp = read_qsym2_yaml::<Input>(&name).unwrap();

    if let SymmetryGroupDetectionInputKind::Parameters(inp_pd_params) =
        inp.symmetry_group_detection.unwrap()
    {
        assert_eq!(inp_pd_params.moi_thresholds, vec![1e-4, 1e-5, 1e-6]);
        assert_eq!(inp_pd_params.distance_thresholds, vec![1e-5, 1e-6]);
        assert!(inp_pd_params.time_reversal);
        assert_eq!(
            inp_pd_params.fictitious_magnetic_fields.as_ref().unwrap()[0].0,
            Point3::new(0.0, 1.0, 0.0)
        );
        assert_eq!(
            inp_pd_params.fictitious_magnetic_fields.as_ref().unwrap()[0].1,
            Vector3::new(1.0, 0.0, 0.0)
        );
        assert!(inp_pd_params.fictitious_electric_fields.is_none());
        assert!(inp_pd_params.field_origin_com);
        assert!(!inp_pd_params.write_symmetry_elements);
        assert!(inp_pd_params.result_save_name.is_none());
    } else {
        assert!(false);
    }

    let inp_rep_params = inp.det_representation_analysis.unwrap();
    assert_eq!(inp_rep_params.integrality_threshold, 1e-8);
    assert_eq!(
        inp_rep_params.linear_independence_threshold,
        1e-7
    );
    assert!(inp_rep_params.analyse_mo_symmetries);
    assert!(!inp_rep_params.use_magnetic_group);
    assert!(!inp_rep_params.use_double_group);
    assert!(matches!(
        inp_rep_params.symmetry_transformation_kind,
        SymmetryTransformationKind::Spatial
    ));
    assert!(matches!(
        inp_rep_params.write_character_table,
        Some(CharacterTableDisplay::Numerical)
    ));
    assert!(inp_rep_params.write_overlap_eigenvalues);
    assert_eq!(
        inp_rep_params.infinite_order_to_finite,
        Some(8)
    );
}

#[test]
fn test_interfaces_input_symmetry_group_detection_fromfile() {
    let name = format!("{ROOT}/tests/input/test_input_symmetry_group_detection_fromfile.yml");
    let inp = read_qsym2_yaml::<Input>(&name).unwrap();

    if let SymmetryGroupDetectionInputKind::FromFile(name) = inp.symmetry_group_detection.unwrap() {
        assert_eq!(name, "test_file")
    } else {
        assert!(false);
    }

    let inp_rep_params = inp.det_representation_analysis.unwrap();
    assert_eq!(inp_rep_params.integrality_threshold, 1e-7);
    assert_eq!(
        inp_rep_params.linear_independence_threshold,
        1e-7
    );
    assert!(inp_rep_params.analyse_mo_symmetries);
    assert!(!inp_rep_params.use_magnetic_group);
    assert!(!inp_rep_params.use_double_group);
    assert!(matches!(
        inp_rep_params.symmetry_transformation_kind,
        SymmetryTransformationKind::Spatial
    ));
    assert!(matches!(
        inp_rep_params.write_character_table,
        Some(CharacterTableDisplay::Symbolic)
    ));
    assert!(inp_rep_params.write_overlap_eigenvalues);
    assert!(inp_rep_params.infinite_order_to_finite.is_none());
}
