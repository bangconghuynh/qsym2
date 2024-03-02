//! Command-line interface for QSym².

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use lazy_static::lazy_static;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use regex::Regex;

use crate::auxiliary::contributors::CONTRIBUTORS;
use crate::io::format::{log_subtitle, log_title, qsym2_output, QSym2Output};

/// The current version of QSym².
const VERSION: Option<&str> = option_env!("CARGO_PKG_VERSION");

// =======
// Structs
// =======

/// Enumerated type for subcommands.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Generates a template YAML configuration file and exits.
    Template {
        /// The name for the generated template YAML configuration file.
        #[arg(short, long)]
        name: Option<PathBuf>,
    },

    /// Runs an analysis calculation and exits.
    Run {
        /// The configuration YAML file specifying parameters for the calculation.
        #[arg(short, long, required = true)]
        config: PathBuf,

        /// The output filename.
        #[arg(short, long, required = true)]
        output: PathBuf,

        /// Turn debugging information on.
        #[arg(short, long, action = clap::ArgAction::Count)]
        debug: u8,
    },
}

/// Structure to handle command-line interface parsing.
#[derive(Parser, Debug)]
#[command(author, version, about)]
#[command(next_line_help = true)]
pub struct Cli {
    /// Subcommands.
    #[command(subcommand)]
    pub command: Commands,
}

impl fmt::Display for Cli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.command {
            Commands::Template { name } => {
                writeln!(
                    f,
                    "Generate a template configuration YAML file: {}",
                    name.as_ref()
                        .map(|name| name.display().to_string())
                        .unwrap_or("no name specified".to_string())
                )
            }
            Commands::Run {
                config,
                output,
                debug,
            } => {
                writeln!(f, "{:<11}: {}", "Config file", config.display().to_string())?;
                writeln!(f, "{:<11}: {}", "Output file", output.display().to_string())?;
                writeln!(f, "{:<11}: {}", "Debug level", debug)?;
                Ok(())
            }
        }
    }
}

// =========
// Functions
// =========

/// Outputs a nicely formatted QSym2 heading to the `qsym2-output` logger.
#[cfg_attr(feature = "python", pyfunction)]
pub fn qsym2_output_heading() {
    let version = if let Some(ver) = VERSION {
        format!("v{ver}")
    } else {
        format!("v unknown")
    };
    // Banner length: 103
    qsym2_output!("╭─────────────────────────────────────────────────────────────────────────────────────────────────────╮");
    qsym2_output!("│                                                                                 222222222222222     │");
    qsym2_output!("│                                                                                2:::::::::::::::22   │");
    qsym2_output!("│                                                                                2::::::222222:::::2  │");
    qsym2_output!("│                                                                                2222222     2:::::2  │");
    qsym2_output!("│                                                                                            2:::::2  │");
    qsym2_output!("│      QQQQQQQQQ        SSSSSSSSSSSSSSS                                                      2:::::2  │");
    qsym2_output!("│    QQ:::::::::QQ    SS:::::::::::::::S                                                  2222::::2   │");
    qsym2_output!("│  QQ:::::::::::::QQ S:::::SSSSSS::::::S                                             22222::::::22    │");
    qsym2_output!("│ Q:::::::QQQ:::::::QS:::::S     SSSSSSS                                           22::::::::222      │");
    qsym2_output!("│ Q::::::O   Q::::::QS:::::S      yyyyyyy           yyyyyyy mmmmmmm    mmmmmmm    2:::::22222         │");
    qsym2_output!("│ Q:::::O     Q:::::QS:::::S       y:::::y         y:::::ymm:::::::m  m:::::::mm 2:::::2              │");
    qsym2_output!("│ Q:::::O     Q:::::Q S::::SSSS     y:::::y       y:::::ym::::::::::mm::::::::::m2:::::2              │");
    qsym2_output!("│ Q:::::O     Q:::::Q  SS::::::SSSSS y:::::y     y:::::y m::::::::::::::::::::::m2:::::2       222222 │");
    qsym2_output!("│ Q:::::O     Q:::::Q    SSS::::::::SSy:::::y   y:::::y  m:::::mmm::::::mmm:::::m2::::::2222222:::::2 │");
    qsym2_output!("│ Q:::::O     Q:::::Q       SSSSSS::::Sy:::::y y:::::y   m::::m   m::::m   m::::m2::::::::::::::::::2 │");
    qsym2_output!("│ Q:::::O  QQQQ:::::Q            S:::::Sy:::::y:::::y    m::::m   m::::m   m::::m22222222222222222222 │");
    qsym2_output!("│ Q::::::O Q::::::::Q            S:::::S y:::::::::y     m::::m   m::::m   m::::m                     │");
    qsym2_output!("│ Q:::::::QQ::::::::QSSSSSSS     S:::::S  y:::::::y      m::::m   m::::m   m::::m                     │");
    qsym2_output!("│  QQ::::::::::::::Q S::::::SSSSSS:::::S   y:::::y       m::::m   m::::m   m::::m                     │");
    qsym2_output!("│    QQ:::::::::::Q  S:::::::::::::::SS   y:::::y        m::::m   m::::m   m::::m                     │");
    qsym2_output!("│      QQQQQQQQ::::QQ SSSSSSSSSSSSSSS    y:::::y         mmmmmm   mmmmmm   mmmmmm                     │");
    qsym2_output!("│              Q:::::Q                  y:::::y                                                       │");
    qsym2_output!("│               QQQQQQ                 y:::::y                A program for Quantum Symbolic Symmetry │");
    qsym2_output!("│                                     y:::::y                                                         │");
    qsym2_output!("│                                    y:::::y                                     {version:>13} (2024) │");
    qsym2_output!("│                                   yyyyyyy                                     Author: Bang C. Huynh │");
    qsym2_output!("╰─────────────────────────────────────────────────────────────────────────────────────────────────────╯");
    qsym2_output!("       developed with support from the ERC's topDFT project at the University of Nottingham, UK");
    qsym2_output!("");
    qsym2_output!("          If you find QSym² helpful in your research, please support us by citing our paper:");
    qsym2_output!("                         Huynh, B. C., Wibowo-Teale, M. & Wibowo-Teale, A. M.");
    qsym2_output!("                              *J. Chem. Theory Comput.* **20**, 114–133.");
    qsym2_output!("                                 doi:10.1021/acs.jctc.3c01118 (2024)");
    qsym2_output!("");
}

lazy_static! {
    /// Regular expression pattern for lines commented out with `#`.
    static ref COMMENT_RE: Regex = Regex::new(r"^\s*#.*?").expect("Regex pattern invalid.");
}

/// Outputs a nicely formatted list of contributors.
#[cfg_attr(feature = "python", pyfunction)]
pub fn qsym2_output_contributors() {
    qsym2_output!("    Contributors (in alphabetical order):");
    CONTRIBUTORS.iter().for_each(|contrib| {
        qsym2_output!("        {}", contrib.trim());
    });
    qsym2_output!("");
}

/// Outputs a summary of the calculation.
///
/// # Arguments
///
/// * `config_path` - The path of the configuration YAML file defining the calculation parameters.
/// * `cli` - The parsed command-line arguments.
pub fn qsym2_output_calculation_summary<P: AsRef<Path>>(config_path: P, cli: &Cli) {
    log_title("Calculation Summary");
    qsym2_output!("");

    log_subtitle("Command line arguments");
    cli.log_output_display();
    qsym2_output!("");

    log_subtitle("Input YAML configuration file");
    let config_contents =
        fs::read_to_string(&config_path).expect("Input configuration YAML file could not be read.");

    qsym2_output!("File path: {}", config_path.as_ref().display());
    let filtered_config_contents = config_contents
        .lines()
        .filter_map(|line| {
            if COMMENT_RE.is_match(line) {
                None
            } else {
                Some(line.trim_end().to_string())
            }
        })
        .collect::<Vec<_>>();
    let length = filtered_config_contents
        .iter()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(20);
    let formatted_config_contents = itertools::intersperse(
        filtered_config_contents
            .iter()
            .map(|line| format!("┊ {line:<length$} ┊")),
        "\n".to_string(),
    )
    .collect::<String>();
    qsym2_output!("┌{}┐", "┄".repeat(length + 2));
    formatted_config_contents.trim().log_output_display();
    qsym2_output!("└{}┘", "┄".repeat(length + 2));
    qsym2_output!("");
    qsym2_output!("");
}
