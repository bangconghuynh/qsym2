use crate::interfaces::input::ao_basis::*;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::SlaterDeterminantRepAnalysisParams;
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionParams;
use crate::interfaces::input::analysis::SlaterDeterminantSourceHandle;
use crate::interfaces::input::SymmetryGroupDetectionInputKind;

use super::{BinariesSlaterDeterminantSource, ByteOrder, MatrixOrder};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_interfaces_binaries_bf3() {
    let pd_params = SymmetryGroupDetectionParams::default();
    let pd_params_inp = SymmetryGroupDetectionInputKind::Parameters(pd_params);
    let afa_params = AngularFunctionRepAnalysisParams::default();
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::default();

    let ibs = vec![
        InputBasisShell::builder()
            .l(0)
            .shell_order(InputShellOrder::PureIncreasingm)
            .build()
            .unwrap(),
        InputBasisShell::builder()
            .l(0)
            .shell_order(InputShellOrder::PureIncreasingm)
            .build()
            .unwrap(),
        InputBasisShell::builder()
            .l(0)
            .shell_order(InputShellOrder::PureIncreasingm)
            .build()
            .unwrap(),
        InputBasisShell::builder()
            .l(0)
            .shell_order(InputShellOrder::PureIncreasingm)
            .build()
            .unwrap(),
        InputBasisShell::builder()
            .l(1)
            .shell_order(InputShellOrder::CartQChem)
            .build()
            .unwrap(),
        InputBasisShell::builder()
            .l(1)
            .shell_order(InputShellOrder::CartQChem)
            .build()
            .unwrap(),
        InputBasisShell::builder()
            .l(1)
            .shell_order(InputShellOrder::CartQChem)
            .build()
            .unwrap(),
        InputBasisShell::builder()
            .l(2)
            .shell_order(InputShellOrder::PureIncreasingm)
            .build()
            .unwrap(),
        InputBasisShell::builder()
            .l(2)
            .shell_order(InputShellOrder::PureIncreasingm)
            .build()
            .unwrap(),
        InputBasisShell::builder()
            .l(3)
            .shell_order(InputShellOrder::PureIncreasingm)
            .build()
            .unwrap(),
    ];
    let ibao = InputBasisAngularOrder(vec![
        InputBasisAtom::builder()
            .atom((0, "B".to_string()))
            .basis_shells(ibs.clone())
            .build()
            .unwrap(),
        InputBasisAtom::builder()
            .atom((1, "F".to_string()))
            .basis_shells(ibs.clone())
            .build()
            .unwrap(),
        InputBasisAtom::builder()
            .atom((2, "F".to_string()))
            .basis_shells(ibs.clone())
            .build()
            .unwrap(),
        InputBasisAtom::builder()
            .atom((3, "F".to_string()))
            .basis_shells(ibs)
            .build()
            .unwrap(),
    ]);

    let bin_sd_source = BinariesSlaterDeterminantSource::builder()
        .xyz(format!("{ROOT}/tests/binaries/bf3_ccsd/xyz").into())
        .sao(format!("{ROOT}/tests/binaries/bf3_ccsd/sao_f").into())
        .sao(format!("{ROOT}/tests/binaries/bf3_ccsd/sao_f").into())
        .coefficients(vec![
            format!("{ROOT}/tests/binaries/bf3_ccsd/ca_f").into(),
            format!("{ROOT}/tests/binaries/bf3_ccsd/cb_f").into(),
        ])
        .occupations(vec![
            format!("{ROOT}/tests/binaries/bf3_ccsd/occa").into(),
            format!("{ROOT}/tests/binaries/bf3_ccsd/occb").into(),
        ])
        .bao(ibao)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .matrix_order(MatrixOrder::ColMajor)
        .byte_order(ByteOrder::LittleEndian)
        .build()
        .unwrap();
    let res = bin_sd_source
        .sd_source_handle(&pd_params_inp, &afa_params, &sda_params)
        .unwrap();
    assert_eq!(res.0, "D3h");
    assert_eq!(res.1, "|A|^(')_(1)");
}
