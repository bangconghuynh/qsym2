//! QSym² interface with binary data files.

use std::path::PathBuf;

use anyhow::{format_err, Context};
use byteorder::{BigEndian, LittleEndian};
use derive_builder::Builder;
use ndarray::{Array1, Array2, Array4, ShapeBuilder};
use serde::{Deserialize, Serialize};

use crate::analysis::Metric;
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::{
    SlaterDeterminantRepAnalysisDriver, SlaterDeterminantRepAnalysisParams,
};
use crate::drivers::representation_analysis::MagneticSymmetryAnalysisKind;
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

    /// Path to a binary file containing the two-centre atomic-orbital overlap matrix. The
    /// dimensions of the matrix will be used to determine whether this is a *spatial* matrix or a
    /// *full* matrix.
    pub sao: PathBuf,

    /// Optional path to a binary file containing the four-centre atomic-orbital overlap matrix.
    /// This is only required for density symmetry analysis. The dimensions of the matrix will be
    /// used to determine whether this is a *spatial* matrix or a *full* matrix.
    #[builder(default = "None")]
    pub sao_4c: Option<PathBuf>,

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

impl BinariesSlaterDeterminantSource {
    /// Returns a builder to construct a structure for handling binaries Slater determinant
    /// source.
    fn builder() -> BinariesSlaterDeterminantSourceBuilder {
        BinariesSlaterDeterminantSourceBuilder::default()
    }
}

impl Default for BinariesSlaterDeterminantSource {
    fn default() -> Self {
        BinariesSlaterDeterminantSource {
            xyz: PathBuf::from("path/to/xyz"),
            sao: PathBuf::from("path/to/2c/ao/overlap/matrix"),
            sao_4c: None,
            coefficients: vec![
                PathBuf::from("path/to/alpha/coeffs"),
                PathBuf::from("path/to/beta/coeffs"),
            ],
            occupations: vec![
                PathBuf::from("path/to/alpha/occupations"),
                PathBuf::from("path/to/beta/occupations"),
            ],
            bao: InputBasisAngularOrder::default(),
            spin_constraint: SpinConstraint::Unrestricted(2, false),
            matrix_order: MatrixOrder::default(),
            byte_order: ByteOrder::default(),
        }
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
        let nspatial = bao.n_funcs();

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
        let sao_2c_arr = match self.matrix_order {
            MatrixOrder::RowMajor => {
                if sao_v.len() == nspatial.pow(2) {
                    Array2::from_shape_vec((nspatial, nspatial), sao_v)
                    .with_context(|| {
                        format!("Unable to construct an AO overlap matrix with dimensions {nspatial} × {nspatial} from the read-in row-major binary file when handling custom Slater determinant source")
                    }).map_err(|err| format_err!(err))
                } else if sao_v.len() == (2 * nspatial).pow(2) {
                    Array2::from_shape_vec((2 * nspatial, 2 * nspatial), sao_v)
                    .with_context(|| {
                        format!("Unable to construct an AO overlap matrix with dimensions {} × {} from the read-in row-major binary file when handling custom Slater determinant source", 2 * nspatial, 2 * nspatial)
                    }).map_err(|err| format_err!(err))
                } else if sao_v.len() == (4 * nspatial).pow(2) {
                    Array2::from_shape_vec((4 * nspatial, 4 * nspatial), sao_v)
                    .with_context(|| {
                        format!("Unable to construct an AO overlap matrix with dimensions {} × {} from the read-in row-major binary file when handling custom Slater determinant source", 4 * nspatial, 4 * nspatial)
                    }).map_err(|err| format_err!(err))
                } else {
                    Err(format_err!("Invalid dimensions of read-in SAO matrix.")).with_context(|| "Unable to construct an AO overlap matrix from the read-in row-major binary file when handling custom Slater determinant source")
                }
            }
            MatrixOrder::ColMajor => {
                if sao_v.len() == nspatial.pow(2) {
                    Array2::from_shape_vec((nspatial, nspatial).f(), sao_v)
                .with_context(|| {
                    format!("Unable to construct an AO overlap matrix with dimensions {nspatial} × {nspatial} from the read-in column-major binary file when handling custom Slater determinant source")
                }).map_err(|err| format_err!(err))
                } else if sao_v.len() == (2 * nspatial).pow(2) {
                    Array2::from_shape_vec((2 * nspatial, 2 * nspatial).f(), sao_v)
                .with_context(|| {
                    format!("Unable to construct an AO overlap matrix with dimensions {} × {} from the read-in column-major binary file when handling custom Slater determinant source", 2 * nspatial, 2 * nspatial)
                }).map_err(|err| format_err!(err))
                } else if sao_v.len() == (4 * nspatial).pow(2) {
                    Array2::from_shape_vec((4 * nspatial, 4 * nspatial).f(), sao_v)
                    .with_context(|| {
                        format!("Unable to construct an AO overlap matrix with dimensions {} × {} from the read-in column-major binary file when handling custom Slater determinant source", 4 * nspatial, 4 * nspatial)
                    }).map_err(|err| format_err!(err))
                } else {
                    Err(format_err!("Invalid dimensions of read-in SAO matrix.")).with_context(|| "Unable to construct an AO overlap matrix from the read-in column-major binary file when handling custom Slater determinant source")
                }
            }
        }?;

        let sao_2c = match self.spin_constraint {
            SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
                if sao_2c_arr.shape() == &[nspatial, nspatial] {
                    Ok(Metric::Spatial(&sao_2c_arr, None))
                } else {
                    Err(format_err!(
                        "Unexpected dimensions for `sao_2c`: {} × {} for {nspatial} spatial basis {} in spin constraint {}.",
                        sao_2c_arr.nrows(),
                        sao_2c_arr.ncols(),
                        if nspatial == 1 {"function"} else {"functions"},
                        self.spin_constraint
                    ))
                }
            }
            SpinConstraint::Generalised(nspins, _) => {
                if sao_2c_arr.shape() == &[nspatial, nspatial] {
                    Ok(Metric::Spatial(&sao_2c_arr, None))
                } else if sao_2c_arr.shape()
                    == &[
                        usize::from(nspins) * nspatial,
                        usize::from(nspins) * nspatial,
                    ]
                {
                    Ok(Metric::Full(&sao_2c_arr, None))
                } else {
                    Err(format_err!(
                        "Unexpected dimensions for `sao_2c`: {} × {} for {nspatial} spatial basis {} in spin constraint {}.",
                        sao_2c_arr.nrows(),
                        sao_2c_arr.ncols(),
                        if nspatial == 1 {"function"} else {"functions"},
                        self.spin_constraint
                    ))
                }
            }
            SpinConstraint::RelativisticGeneralised(nspins, _, _) => {
                if sao_2c_arr.shape() == &[nspatial, nspatial] {
                    Ok(Metric::Spatial(&sao_2c_arr, None))
                } else if sao_2c_arr.shape()
                    == &[
                        2 * usize::from(nspins) * nspatial,
                        2 * usize::from(nspins) * nspatial,
                    ]
                {
                    Ok(Metric::Full(&sao_2c_arr, None))
                } else {
                    Err(format_err!(
                        "Unexpected dimensions for `sao_2c`: {} × {} for {nspatial} spatial basis {} in spin constraint {}.",
                        sao_2c_arr.nrows(),
                        sao_2c_arr.ncols(),
                        if nspatial == 1 {"function"} else {"functions"},
                        self.spin_constraint
                    ))
                }
            }
        }?;

        let sao_4c_arr_opt = if let Some(sao_4c_path) = self.sao_4c.as_ref() {
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
                MatrixOrder::RowMajor => Array4::from_shape_vec((nspatial, nspatial, nspatial, nspatial), sao_4c_v)
                    .with_context(|| {
                        "Unable to construct a four-centre AO overlap matrix from the read-in row-major binary file when handling custom Slater determinant source"
                    })?,
                MatrixOrder::ColMajor => Array4::from_shape_vec((nspatial, nspatial, nspatial, nspatial).f(), sao_4c_v)
                    .with_context(|| {
                        "Unable to construct a four-centre AO overlap matrix from the read-in column-major binary file when handling custom Slater determinant source"
                    })?,
            };
            Some(sao_4c)
        } else {
            None
        };

        let sao_4c = sao_4c_arr_opt
            .as_ref()
            .map(|sao_4c_arr| Metric::Spatial(sao_4c_arr, None));

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
                    let nmo = c_v.len().div_euclid(nspatial);
                    Array2::from_shape_vec((nspatial, nmo), c_v)
                })
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| {
                    "Unable to construct coefficient matrix (matrices) from the read-in row-major binary file(s) when handling custom Slater determinant source"
                })?,

            MatrixOrder::ColMajor => cs_v
                .into_iter()
                .map(|c_v| {
                    let nmo = c_v.len().div_euclid(nspatial);
                    Array2::from_shape_vec((nspatial, nmo).f(), c_v)
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
            Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                    MagneticRepresentedSymmetryGroup,
                    f64,
                >::builder()
                .parameters(sda_params)
                .angular_function_parameters(afa_params)
                .determinant(&det)
                .sao_2c(sao_2c)
                .sao_4c(sao_4c)
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
                >::builder()
                .parameters(sda_params)
                .angular_function_parameters(afa_params)
                .determinant(&det)
                .sao_2c(sao_2c)
                .sao_4c(sao_4c)
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

/// Enumerated type indicating the byte order of numerical values in binary files.
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
