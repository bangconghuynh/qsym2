use std::fs::File;
use std::io::{BufReader, BufWriter};

use anyhow::{self, format_err};
use bincode;
use serde::{de::DeserializeOwned, Serialize};
use serde_yaml;

/// An enumerated type for `QSym2` file types.
pub enum QSym2FileType {
    /// Variant for binary files containing symmetry-group detection results.
    Sym,

    /// Variant for binary files containing symmetry groups.
    Grp,

    /// Variant for binary files containing representation analysis results.
    Rep,

    /// Variant for binary files containing character tables.
    Chr,
}

impl QSym2FileType {
    /// Returns the extension of the file type.
    pub fn ext(&self) -> String {
        match self {
            QSym2FileType::Sym => ".qsym2.sym".to_string(),
            QSym2FileType::Grp => ".qsym2.grp".to_string(),
            QSym2FileType::Rep => ".qsym2.rep".to_string(),
            QSym2FileType::Chr => ".qsym2.chr".to_string(),
        }
    }
}

/// Reads a `QSym2` binary file and deserialises it into an appropriate structure.
///
/// # Arguments
///
/// * `name` - The name of the file to be read in (without `QSym2`-specific extensions).
/// * `file_type` - The type of the `QSym2` file to be read in.
///
/// # Returns
///
/// A `Result` containing the structure deserialised from the read-in file.
pub fn read_qsym2_binary<T>(name: &str, file_type: QSym2FileType) -> Result<T, anyhow::Error>
where
    T: DeserializeOwned,
{
    let file_name = format!("{name}{}", file_type.ext());
    let mut reader = BufReader::new(File::open(file_name).map_err(|err| format_err!(err))?);
    bincode::deserialize_from(&mut reader).map_err(|err| format_err!(err))
}

/// Serialises a structure and writes into a `QSym2` binary file.
///
/// # Arguments
///
/// * `name` - The name of the file to be written (without `QSym2`-specific extensions).
/// * `file_type` - The type of the `QSym2` file to be written.
///
/// # Returns
///
/// A `Result` indicating if the serialisation and writing processes have been successful.
pub fn write_qsym2_binary<T>(
    name: &str,
    file_type: QSym2FileType,
    value: &T,
) -> Result<(), anyhow::Error>
where
    T: Serialize,
{
    let file_name = format!("{name}{}", file_type.ext());
    let mut writer = BufWriter::new(File::create(file_name)?);
    bincode::serialize_into(&mut writer, value).map_err(|err| format_err!(err))
}

/// Reads a `QSym2` configuration YAML file and deserialises it into an appropriate structure.
///
/// # Arguments
///
/// * `name` - The name of the file to be read in (with its `.yml` or `.yaml` extension).
///
/// # Returns
///
/// A `Result` containing the structure deserialised from the read-in file.
pub fn read_qsym2_yaml<T>(name: &str) -> Result<T, anyhow::Error>
where
    T: DeserializeOwned,
{
    let mut reader = BufReader::new(File::open(name).map_err(|err| format_err!(err))?);
    serde_yaml::from_reader(&mut reader).map_err(|err| format_err!(err))
}

/// Serialises a structure and writes into a `QSym2` configuration YAML file.
///
/// # Arguments
///
/// * `name` - The name of the YAML file to be written (without extensions). The resulting file
/// will have the `.yml` extension.
///
/// # Returns
///
/// A `Result` indicating if the serialisation and writing processes have been successful.
pub fn write_qsym2_yaml<T>(
    name: &str,
    value: &T,
) -> Result<(), anyhow::Error>
where
    T: Serialize,
{
    let file_name = format!("{name}.yml");
    let mut writer = BufWriter::new(File::create(file_name)?);
    serde_yaml::to_writer(&mut writer, value).map_err(|err| format_err!(err))
}
