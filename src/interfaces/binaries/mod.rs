//! QSym² interface with binary data files.

use std::path::PathBuf;

use anyhow::{Context, format_err};
use byteorder::{BigEndian, LittleEndian};
use derive_builder::Builder;
use itertools::Itertools;
use ndarray::{Array1, Array2, Array4, ShapeBuilder};
use serde::{Deserialize, Serialize};

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::drivers::QSym2Driver;
use crate::drivers::representation_analysis::MagneticSymmetryAnalysisKind;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::{
    SlaterDeterminantRepAnalysisDriver, SlaterDeterminantRepAnalysisParams,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionDriver;
use crate::interfaces::input::SymmetryGroupDetectionInputKind;
use crate::interfaces::input::analysis::SlaterDeterminantSourceHandle;
use crate::interfaces::input::ao_basis::InputBasisAngularOrder;
use crate::io::numeric::NumericReader;
use crate::io::{QSym2FileType, read_qsym2_binary};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};
use crate::target::determinant::SlaterDeterminant;

#[cfg(test)]
#[path = "binaries_tests.rs"]
mod binaries_tests;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Input target: Slater determinant; source: binaries
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Serialisable/deserialisable structure containing control parameters for acquiring Slater
/// determinant(s) from a custom specification.
#[derive(Clone, Builder, Serialize, Deserialize)]
pub struct BinariesSlaterDeterminantSource {
    /// Path to an XYZ file containing the molecular geometry.
    pub xyz: PathBuf,

    /// Path to a binary file containing the two-centre atomic-orbital spatial overlap matrix.
    pub sao: PathBuf,

    /// Optional path to a binary file containing the four-centre atomic-orbital spatial overlap
    /// matrix. This is only required for density symmetry analysis.
    #[builder(default = "None")]
    pub sao_4c: Option<PathBuf>,

    /// Paths to binary files containing molecular-orbital coefficient matrices for different spin
    /// spaces.
    pub coefficients: Vec<PathBuf>,

    /// Paths to binary files containing occupation numbers for the molecular orbitals.
    pub occupations: Vec<PathBuf>,

    /// Specifications of basis angular order information, one for each explicit component per
    /// coefficient matrix.
    pub baos: Vec<InputBasisAngularOrder>,

    /// Specification of spin constraint.
    pub spin_constraint: SpinConstraint,

    /// Specification of the order matrix elements are packed in binary files.
    pub matrix_order: MatrixOrder,

    /// Specification of the byte order numerical values are stored in binary files.
    pub byte_order: ByteOrder,
}

impl BinariesSlaterDeterminantSource {
    /// Returns a builder to construct a structure for handling binaries Slater determinant
    /// source.
    pub fn builder() -> BinariesSlaterDeterminantSourceBuilder {
        BinariesSlaterDeterminantSourceBuilder::default()
    }
}

impl Default for BinariesSlaterDeterminantSource {
    fn default() -> Self {
        BinariesSlaterDeterminantSource::builder()
            .xyz(PathBuf::from("path/to/xyz"))
            .sao(PathBuf::from("path/to/2c/ao/overlap/matrix"))
            .sao_4c(None)
            .coefficients(vec![
                PathBuf::from("path/to/alpha/coeffs"),
                PathBuf::from("path/to/beta/coeffs"),
            ])
            .occupations(vec![
                PathBuf::from("path/to/alpha/occupations"),
                PathBuf::from("path/to/beta/occupations"),
            ])
            .baos(vec![InputBasisAngularOrder::default()])
            .spin_constraint(SpinConstraint::Unrestricted(2, false))
            .matrix_order(MatrixOrder::default())
            .byte_order(ByteOrder::default())
            .build()
            .expect("Unable to build a default `BinariesSlaterDeterminantSource`.")
    }
}

impl SlaterDeterminantSourceHandle for BinariesSlaterDeterminantSource {
    type Outcome = (String, String);

    fn sd_source_handle(
        &self,
        pd_params_inp: &SymmetryGroupDetectionInputKind,
        afa_params: &AngularFunctionRepAnalysisParams,
        sda_params: &SlaterDeterminantRepAnalysisParams<f64>,
    ) -> Result<Self::Outcome, anyhow::Error> {
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
            SymmetryGroupDetectionInputKind::FromFile(pd_res_file) => read_qsym2_binary(
                pd_res_file,
                QSym2FileType::Sym,
            )
            .with_context(|| {
                format!(
                    "Unable to read `{}.qsym2.sym` when handling custom Slater determinant source",
                    pd_res_file.display()
                )
            })?,
        };
        let mol = &pd_res.pre_symmetry.recentred_molecule;
        let baos = self.baos.iter().map(|bao| bao.to_basis_angular_order(mol)
            .with_context(|| "Unable to digest the input basis angular order information when handling custom Slater determinant source")).collect::<Result<Vec<_>, _>>()?;
        let nfuncs_tot = baos.iter().map(|bao| bao.n_funcs()).sum::<usize>();

        let sao_v = match self.byte_order {
            ByteOrder::LittleEndian => {
                NumericReader::<_, LittleEndian, f64>::from_file(&self.sao)
                    .with_context(|| {
                        "Unable to read the specified two-centre SAO file when handling custom Slater determinant source"
                    })?.collect::<Vec<_>>()
            }
            ByteOrder::BigEndian => {
                NumericReader::<_, BigEndian, f64>::from_file(&self.sao)
                    .with_context(|| {
                        "Unable to read the specified two-centre SAO file when handling custom Slater determinant source"
                    })?.collect::<Vec<_>>()
            }
        };
        let sao = match self.matrix_order {
            MatrixOrder::RowMajor => Array2::from_shape_vec((nfuncs_tot, nfuncs_tot), sao_v)
                .with_context(|| {
                    "Unable to construct an AO overlap matrix from the read-in row-major binary file when handling custom Slater determinant source"
                })?,
            MatrixOrder::ColMajor => Array2::from_shape_vec((nfuncs_tot, nfuncs_tot).f(), sao_v)
                .with_context(|| {
                    "Unable to construct an AO overlap matrix from the read-in column-major binary file when handling custom Slater determinant source"
                })?,
        };

        let sao_4c = if let Some(sao_4c_path) = self.sao_4c.as_ref() {
            let sao_4c_v = match self.byte_order {
                ByteOrder::LittleEndian => {
                    NumericReader::<_, LittleEndian, f64>::from_file(sao_4c_path)
                        .with_context(|| {
                            "Unable to read the specified four-centre SAO file when handling custom Slater determinant source"
                        })?.collect::<Vec<_>>()
                }
                ByteOrder::BigEndian => {
                    NumericReader::<_, BigEndian, f64>::from_file(sao_4c_path)
                        .with_context(|| {
                            "Unable to read the specified four-centre SAO file when handling custom Slater determinant source"
                        })?.collect::<Vec<_>>()
                }
            };
            let sao_4c = match self.matrix_order {
                MatrixOrder::RowMajor => Array4::from_shape_vec((nfuncs_tot, nfuncs_tot, nfuncs_tot, nfuncs_tot), sao_4c_v)
                    .with_context(|| {
                        "Unable to construct a four-centre AO overlap matrix from the read-in row-major binary file when handling custom Slater determinant source"
                    })?,
                MatrixOrder::ColMajor => Array4::from_shape_vec((nfuncs_tot, nfuncs_tot, nfuncs_tot, nfuncs_tot).f(), sao_4c_v)
                    .with_context(|| {
                        "Unable to construct a four-centre AO overlap matrix from the read-in column-major binary file when handling custom Slater determinant source"
                    })?,
            };
            Some(sao_4c)
        } else {
            None
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
                    let nmo = c_v.len().div_euclid(nfuncs_tot);
                    Array2::from_shape_vec((nfuncs_tot, nmo), c_v)
                })
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| {
                    "Unable to construct coefficient matrix (matrices) from the read-in row-major binary file(s) when handling custom Slater determinant source"
                })?,

            MatrixOrder::ColMajor => cs_v
                .into_iter()
                .map(|c_v| {
                    let nmo = c_v.len().div_euclid(nfuncs_tot);
                    Array2::from_shape_vec((nfuncs_tot, nmo).f(), c_v)
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

        let det = SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&cs)
            .occupations(&occs)
            .baos(baos.iter().collect_vec())
            .mol(mol)
            .structure_constraint(self.spin_constraint.clone())
            .complex_symmetric(false)
            .threshold(sda_params.linear_independence_threshold)
            .build()
            .with_context(|| "Failed to construct a Slater determinant when handling custom Slater determinant source")?;

        match &sda_params.use_magnetic_group {
            Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                    MagneticRepresentedSymmetryGroup,
                    f64,
                    SpinConstraint,
                >::builder()
                .parameters(sda_params)
                .angular_function_parameters(afa_params)
                .determinant(&det)
                .sao(&sao)
                .sao_spatial_4c(sao_4c.as_ref())
                .symmetry_group(&pd_res)
                .build()
                .with_context(|| {
                    "Failed to construct a Slater determinant corepresentation analysis driver when handling custom Slater determinant source"
                })?;
                sda_driver
                    .run()
                    .with_context(|| {
                        "Failed to execute the Slater determinant corepresentation analysis driver successfully when handling custom Slater determinant source"
                    })?;
                let group_name = pd_res
                    .magnetic_symmetry
                    .as_ref()
                    .and_then(|magsym| magsym.group_name.clone())
                    .ok_or(format_err!("Magnetic group name not found when handling custom Slater determinant source."))?;
                let sym = sda_driver
                    .result()
                    .with_context(|| {
                        "Failed to obtain corepresentation analysis result when handling custom Slater determinant source"
                    })?
                    .determinant_symmetry()
                    .as_ref()
                    .map_err(|err| format_err!(err.clone()))
                    .with_context(|| {
                        "Failed to obtain determinant symmetry from corepresentation analysis result when handling custom Slater determinant source"
                    })?
                    .to_string();
                Ok((group_name, sym))
            }
            Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                    UnitaryRepresentedSymmetryGroup,
                    f64,
                    SpinConstraint,
                >::builder()
                .parameters(sda_params)
                .angular_function_parameters(afa_params)
                .determinant(&det)
                .sao(&sao)
                .sao_spatial_4c(sao_4c.as_ref())
                .symmetry_group(&pd_res)
                .build()
                .with_context(|| {
                    "Failed to construct a Slater determinant representation analysis driver when handling custom Slater determinant source"
                })?;
                sda_driver
                    .run()
                    .with_context(|| {
                        "Failed to execute the Slater determinant representation analysis driver successfully when handling custom Slater determinant source"
                    })?;
                let group_name = if sda_params.use_magnetic_group.is_none() {
                    pd_res
                        .unitary_symmetry
                        .group_name
                        .as_ref()
                        .ok_or(format_err!("Unitary group name not found when handling custom Slater determinant source."))?.clone()
                } else {
                    pd_res
                        .magnetic_symmetry
                        .as_ref()
                        .and_then(|magsym| magsym.group_name.clone())
                        .ok_or(format_err!("Magnetic group name not found when handling custom Slater determinant source."))?
                };
                let sym = sda_driver
                    .result()
                    .with_context(|| {
                        "Failed to obtain representation analysis result when handling custom Slater determinant source"
                    })?
                    .determinant_symmetry()
                    .as_ref()
                    .map_err(|err| format_err!(err.clone()))
                    .with_context(|| {
                        "Failed to obtain determinant symmetry from representation analysis result when handling custom Slater determinant source"
                    })?
                    .to_string();
                Ok((group_name, sym))
            }
        }
    }
}

/// Enumerated type indicating the order the matrix elements are traversed when stored into or
/// read in from a binary file.
#[derive(Clone, Serialize, Deserialize, Default)]
pub enum MatrixOrder {
    #[default]
    RowMajor,
    ColMajor,
}

/// Enumerated type indicating the byte order of numerical values in binary files.
#[derive(Clone, Serialize, Deserialize, Default)]
pub enum ByteOrder {
    #[default]
    LittleEndian,
    BigEndian,
}
