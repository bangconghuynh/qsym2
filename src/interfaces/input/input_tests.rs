use nalgebra::{Point3, Vector3};

use crate::drivers::representation_analysis::CharacterTableDisplay;
use crate::io::read_qsym2_yaml;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;

use super::{AnalysisTarget, Input, SymmetryGroupDetectionInputKind};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_interfaces_input_symmetry_group_detection_parameters() {
    let name = format!("{ROOT}/tests/input/test_input_symmetry_group_detection_parameters.yml");
    let inp = read_qsym2_yaml::<Input, _>(&name).unwrap();

    if let SymmetryGroupDetectionInputKind::Parameters(inp_pd_params) = inp.symmetry_group_detection
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

    if let AnalysisTarget::SlaterDeterminant(sd_control) = &inp.analysis_targets[0] {
        let inp_rep_params = &sd_control.control;
        assert_eq!(inp_rep_params.integrality_threshold, 1e-8);
        assert_eq!(inp_rep_params.linear_independence_threshold, 1e-7);
        assert!(inp_rep_params.analyse_mo_symmetries);
        assert!(inp_rep_params.use_magnetic_group.is_none());
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
        assert_eq!(inp_rep_params.infinite_order_to_finite, Some(8));
    } else {
        assert!(false);
    }
}

#[test]
fn test_interfaces_input_symmetry_group_detection_fromfile() {
    let name = format!("{ROOT}/tests/input/test_input_symmetry_group_detection_fromfile.yml");
    let inp = read_qsym2_yaml::<Input, _>(&name).unwrap();

    if let SymmetryGroupDetectionInputKind::FromFile(name) = inp.symmetry_group_detection {
        assert_eq!(name.to_str().unwrap(), "test_file")
    } else {
        assert!(false);
    }

    if let AnalysisTarget::SlaterDeterminant(sd_control) = &inp.analysis_targets[0] {
        let inp_rep_params = &sd_control.control;
        assert_eq!(inp_rep_params.integrality_threshold, 1e-7);
        assert_eq!(inp_rep_params.linear_independence_threshold, 1e-7);
        assert!(inp_rep_params.analyse_mo_symmetries);
        assert!(inp_rep_params.use_magnetic_group.is_none());
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
    } else {
        assert!(false);
    }
}

#[test]
fn test_interfaces_input_bao() {
    use super::analysis::SlaterDeterminantSource;
    use crate::auxiliary::molecule::Molecule;

    let name = format!("{ROOT}/tests/input/test_input_bao.yml");
    let xyz = format!("{ROOT}/tests/xyz/water.xyz");
    let inp = read_qsym2_yaml::<Input, _>(&name).unwrap();
    let mol = Molecule::from_xyz(&xyz, 1e-7);

    if let AnalysisTarget::SlaterDeterminant(sd_control) = &inp.analysis_targets[0] {
        if let SlaterDeterminantSource::Binaries(binaries_source) = &sd_control.source {
            let bao = binaries_source.bao.to_basis_angular_order(&mol).unwrap();
            assert_eq!(bao.n_funcs(), 41);
            assert_eq!(
                bao.basis_shells()
                    .skip(3)
                    .next()
                    .unwrap()
                    .shell_order
                    .to_string(),
                "Cart (u) (xxx, xxy, xyy, yyy, xxz, xyz, yyz, xzz, yzz, zzz)"
            );
            assert_eq!(
                bao.basis_shells()
                    .skip(5)
                    .next()
                    .unwrap()
                    .shell_order
                    .to_string(),
                "Cart (g) (xx, xy, yy, xz, yz, zz)"
            );
            assert_eq!(
                bao.basis_shells()
                    .skip(7)
                    .next()
                    .unwrap()
                    .shell_order
                    .to_string(),
                "Cart (g) (xx, xy, xz, yy, yz, zz)"
            );
            assert_eq!(
                bao.basis_shells()
                    .skip(8)
                    .next()
                    .unwrap()
                    .shell_order
                    .to_string(),
                "Pure (u) (0, 1, -1, 2, -2, 3, -3)"
            );
        } else {
            assert!(false);
        }
    } else {
        assert!(false);
    };
}

#[test]
fn test_interfaces_input_bao_spinor() {
    use super::analysis::SlaterDeterminantSource;
    use crate::auxiliary::molecule::Molecule;

    let name = format!("{ROOT}/tests/input/test_input_bao_spinor.yml");
    let xyz = format!("{ROOT}/tests/xyz/water.xyz");
    let inp = read_qsym2_yaml::<Input, _>(&name).unwrap();
    let mol = Molecule::from_xyz(&xyz, 1e-7);

    if let AnalysisTarget::SlaterDeterminant(sd_control) = &inp.analysis_targets[0] {
        if let SlaterDeterminantSource::Binaries(binaries_source) = &sd_control.source {
            let bao = binaries_source.bao.to_basis_angular_order(&mol).unwrap();
            assert_eq!(bao.n_funcs(), 16);
            assert_eq!(
                bao.basis_shells().next().unwrap().shell_order.to_string(),
                "Spinor (g) (-1/2, 1/2)"
            );
            assert_eq!(
                bao.basis_shells()
                    .skip(1)
                    .next()
                    .unwrap()
                    .shell_order
                    .to_string(),
                "Spinor (u) (3/2, 1/2, -1/2, -3/2)"
            );
            assert_eq!(
                bao.basis_shells()
                    .skip(2)
                    .next()
                    .unwrap()
                    .shell_order
                    .to_string(),
                "Spinor (g) (1/2, -1/2, 3/2, -3/2, 5/2, -5/2)"
            );
            assert_eq!(
                bao.basis_shells()
                    .skip(3)
                    .next()
                    .unwrap()
                    .shell_order
                    .to_string(),
                "Spinor (u) (-1/2, 1/2)"
            );
            assert_eq!(
                bao.basis_shells()
                    .skip(4)
                    .next()
                    .unwrap()
                    .shell_order
                    .to_string(),
                "Spinor (u) (-1/2, 1/2)"
            );
        } else {
            assert!(false);
        }
    } else {
        assert!(false);
    };
}
