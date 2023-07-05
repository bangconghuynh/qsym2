use std::path::PathBuf;

use anyhow::Context;
use byteorder::{BigEndian, LittleEndian};
use ndarray::{Array1, Array2, ShapeBuilder};
use serde::{Deserialize, Serialize};

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::{
    SlaterDeterminantRepAnalysisDriver, SlaterDeterminantRepAnalysisParams,
};
use crate::drivers::representation_analysis::MagneticSymmetryKind;
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionDriver;
use crate::drivers::QSym2Driver;
use crate::interfaces::input::analysis::SlaterDeterminantSourceHandle;
use crate::interfaces::input::ao_basis::InputBasisAngularOrder;
use crate::interfaces::input::SymmetryGroupDetectionInputKind;
use crate::io::numeric::NumericReader;
use crate::io::{read_qsym2_binary, QSym2FileType};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};
use crate::target::determinant::SlaterDeterminant;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Input target: Slater determinant; source: custom
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A serialisable/deserialisable structure containing control parameters for acquiring Slater
/// determinant(s) from a custom specification.
#[derive(Clone, Serialize, Deserialize)]
pub struct CustomSlaterDeterminantSource {
    /// Path to an XYZ file containing the molecular geometry.
    pub xyz: PathBuf,

    /// Path to a binary file containing the atomic-orbital spatial overlap matrix.
    pub sao: PathBuf,

    /// Paths to binary files containing molecular-orbital coefficient matrices for different spin
    /// spaces.
    pub coefficients: Vec<PathBuf>,

    /// Paths to binary files containing occupation numbers for the molecular orbitals.
    pub occupations: Vec<PathBuf>,

    /// Specification of basis angular order information.
    pub bao: InputBasisAngularOrder,

    /// Specification of spin constraint.
    pub spin_constraint: SpinConstraint,

    /// Specification of the order matrix elements are packed in binary files.
    pub matrix_order: MatrixOrder,

    /// Specification of the byte order numerical values are stored in binary files.
    pub byte_order: ByteOrder,
}

impl Default for CustomSlaterDeterminantSource {
    fn default() -> Self {
        CustomSlaterDeterminantSource {
            xyz: PathBuf::default(),
            sao: PathBuf::default(),
            coefficients: vec![PathBuf::default(), PathBuf::default()],
            occupations: vec![PathBuf::default(), PathBuf::default()],
            bao: InputBasisAngularOrder::default(),
            spin_constraint: SpinConstraint::Unrestricted(2, false),
            matrix_order: MatrixOrder::default(),
            byte_order: ByteOrder::default(),
        }
    }
}

impl SlaterDeterminantSourceHandle for CustomSlaterDeterminantSource {
    fn sd_source_handle(
        &self,
        pd_params_inp: &SymmetryGroupDetectionInputKind,
        afa_params: &AngularFunctionRepAnalysisParams,
        sda_params: &SlaterDeterminantRepAnalysisParams<f64>,
    ) -> Result<(), anyhow::Error> {
        let pd_res = match pd_params_inp {
            SymmetryGroupDetectionInputKind::Parameters(pd_params) => {
                let mut pd_driver = SymmetryGroupDetectionDriver::builder()
                    .parameters(pd_params)
                    .xyz(Some(self.xyz.clone()))
                    .build()
                    .with_context(|| "Unable to construct a symmetry-group detection driver when handling custom Slater determinant source")?;
                pd_driver.run().with_context(|| {
                    "Unable to run the symmetry-group detection driver successfully when handling custom Slater determinant source"
                })?;
                pd_driver
                    .result()
                    .with_context(|| "Unable to retrieve the symmetry-group detection result when handling custom Slater determinant source")?
                    .clone()
            }
            SymmetryGroupDetectionInputKind::FromFile(pd_res_file) => {
                read_qsym2_binary(pd_res_file, QSym2FileType::Sym).with_context(|| {
                    format!(
                    "Unable to read `{}.qsym2.sym` when handling custom Slater determinant source",
                    pd_res_file.display()
                )
                })?
            }
        };
        let mol = &pd_res.pre_symmetry.recentred_molecule;
        let bao = self.bao.to_basis_angular_order(mol)
            .with_context(|| "Unable to digest the input basis angular order information when handling custom Slater determinant source")?;
        let n_spatial = bao.n_funcs();

        let sao_v = match self.byte_order {
            ByteOrder::LittleEndian => {
                NumericReader::<_, LittleEndian, f64>::from_file(&self.sao)
                    .with_context(|| {
                        "Unable to read the specified SAO file when handling custom Slater determinant source"
                    })?.collect::<Vec<_>>()
            }
            ByteOrder::BigEndian => {
                NumericReader::<_, BigEndian, f64>::from_file(&self.sao)
                    .with_context(|| {
                        "Unable to read the specified SAO file when handling custom Slater determinant source"
                    })?.collect::<Vec<_>>()
            }
        };
        let sao = match self.matrix_order {
            MatrixOrder::RowMajor => Array2::from_shape_vec((n_spatial, n_spatial), sao_v)
                .with_context(|| {
                    "Unable to construct an AO overlap matrix from the read-in row-major binary file when handling custom Slater determinant source"
                })?,
            MatrixOrder::ColMajor => Array2::from_shape_vec((n_spatial, n_spatial).f(), sao_v)
                .with_context(|| {
                    "Unable to construct an AO overlap matrix from the read-in column-major binary file when handling custom Slater determinant source"
                })?,
        };

        let cs_v = match self.byte_order {
            ByteOrder::LittleEndian => self
                .coefficients
                .iter()
                .map(|c_path| {
                    NumericReader::<_, LittleEndian, f64>::from_file(c_path)
                        .map(|r| r.collect::<Vec<_>>())
                })
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| {
                    "Unable to read the specified coefficient binary file(s) when handling custom Slater determinant source"
                })?,
            ByteOrder::BigEndian => self
                .coefficients
                .iter()
                .map(|c_path| {
                    NumericReader::<_, BigEndian, f64>::from_file(c_path)
                        .map(|r| r.collect::<Vec<_>>())
                })
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| {
                    "Unable to read the specified coefficient binary file(s) when handling custom Slater determinant source"
                })?,
        };
        let cs = match self.matrix_order {
            MatrixOrder::RowMajor => cs_v
                .into_iter()
                .map(|c_v| {
                    let nmo = c_v.len().div_euclid(n_spatial);
                    Array2::from_shape_vec((n_spatial, nmo), c_v)
                })
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| {
                    "Unable to construct coefficient matrix (matrices) from the read-in row-major binary file(s) when handling custom Slater determinant source"
                })?,

            MatrixOrder::ColMajor => cs_v
                .into_iter()
                .map(|c_v| {
                    let nmo = c_v.len().div_euclid(n_spatial);
                    Array2::from_shape_vec((n_spatial, nmo).f(), c_v)
                })
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| {
                    "Unable to construct coefficient matrix (matrices) from the read-in column-major binary file(s) when handling custom Slater determinant source"
                })?,
        };

        let occs = match self.byte_order {
            ByteOrder::LittleEndian => self
                .occupations
                .iter()
                .map(|occ_path| {
                    Ok::<_, anyhow::Error>(Array1::from_vec(
                        NumericReader::<_, LittleEndian, f64>::from_file(occ_path)
                            .map(|r| r.collect::<Vec<f64>>())?,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| {
                    "Unable to read the specified occupation binary file(s) when handling custom Slater determinant source"
                })?,
            ByteOrder::BigEndian => self
                .occupations
                .iter()
                .map(|occ_path| {
                    Ok::<_, anyhow::Error>(Array1::from_vec(
                        NumericReader::<_, BigEndian, f64>::from_file(occ_path)
                            .map(|r| r.collect::<Vec<f64>>())?,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| {
                    "Unable to read occupation binary file(s) when handling custom Slater determinant source"
                })?,
        };

        let det = SlaterDeterminant::<f64>::builder()
            .coefficients(&cs)
            .occupations(&occs)
            .bao(&bao)
            .mol(mol)
            .spin_constraint(self.spin_constraint.clone())
            .complex_symmetric(false)
            .threshold(sda_params.linear_independence_threshold)
            .build()
            .with_context(|| "Failed to construct a Slater determinant when handling custom Slater determinant source")?;

        match &sda_params.use_magnetic_group {
            Some(MagneticSymmetryKind::Corepresentation) => {
                let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                    MagneticRepresentedSymmetryGroup,
                    f64,
                >::builder()
                .parameters(sda_params)
                .angular_function_parameters(afa_params)
                .determinant(&det)
                .sao_spatial(&sao)
                .symmetry_group(&pd_res)
                .build()
                .with_context(|| {
                    "Failed to construct a Slater determinant corepresentation analysis driver when handling custom Slater determinant source"
                })?;
                sda_driver.run()
            }
            Some(MagneticSymmetryKind::Representation) | None => {
                let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                    UnitaryRepresentedSymmetryGroup,
                    f64,
                >::builder()
                .parameters(sda_params)
                .angular_function_parameters(afa_params)
                .determinant(&det)
                .sao_spatial(&sao)
                .symmetry_group(&pd_res)
                .build()
                .with_context(|| {
                    "Failed to construct a Slater determinant representation analysis driver when handling custom Slater determinant source"
                })?;
                sda_driver.run()
            }
        }
    }
}

/// An enumerated type indicating the order the matrix elements are traversed when stored into or
/// read in from a binary file.
#[derive(Clone, Serialize, Deserialize)]
pub enum MatrixOrder {
    RowMajor,
    ColMajor,
}

impl Default for MatrixOrder {
    fn default() -> Self {
        MatrixOrder::RowMajor
    }
}

/// An enumerated type indicating the byte order of numerical values in binary files.
#[derive(Clone, Serialize, Deserialize)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

impl Default for ByteOrder {
    fn default() -> Self {
        ByteOrder::LittleEndian
    }
}
