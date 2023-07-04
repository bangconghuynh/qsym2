use std::fmt;

use log;

/// Logs an error to the `qsym2-output` logger.
macro_rules! qsym2_error {
    ($fmt:expr $(, $($arg:tt)*)?) => { log::error!(target: "qsym2-output", $fmt, $($($arg)*)?); }
}

/// Logs a warning to the `qsym2-output` logger.
macro_rules! qsym2_warn {
    ($fmt:expr $(, $($arg:tt)*)?) => { log::warn!(target: "qsym2-output", $fmt, $($($arg)*)?); }
}

/// Logs a main output line to the `qsym2-output` logger.
macro_rules! qsym2_output {
    ($fmt:expr $(, $($arg:tt)*)?) => { log::info!(target: "qsym2-output", $fmt, $($($arg)*)?); }
}

pub(crate) use {qsym2_output, qsym2_warn, qsym2_error};

/// Writes a nicely formatted section title.
pub(crate) fn write_title(f: &mut fmt::Formatter<'_>, title: &str) -> fmt::Result {
    let length = title.chars().count();
    let bar = "─".repeat(length);
    writeln!(f, "┌──{bar}──┐")?;
    writeln!(f, "│§ {title} §│")?;
    writeln!(f, "└──{bar}──┘")?;
    Ok(())
}

/// Logs a nicely formatted section title to the `qsym2-output` logger.
pub(crate) fn log_title(title: &str) {
    let length = title.chars().count();
    let bar = "─".repeat(length);
    qsym2_output!("┌──{}──┐", bar);
    qsym2_output!("│§ {} §│", title);
    qsym2_output!("└──{}──┘", bar);
}

/// Writes a nicely formatted subtitle.
pub(crate) fn write_subtitle(f: &mut fmt::Formatter<'_>, subtitle: &str) -> fmt::Result {
    let length = subtitle.chars().count();
    let bar = "═".repeat(length);
    writeln!(f, "{subtitle}")?;
    writeln!(f, "{bar}")?;
    Ok(())
}

/// Logs a nicely formatted subtitle to the `qsym2-output` logger.
pub(crate) fn log_subtitle(subtitle: &str) {
    let length = subtitle.chars().count();
    let bar = "═".repeat(length);
    qsym2_output!("{}", subtitle);
    qsym2_output!("{}", bar);
}

/// Logs a nicely formatted macro-section beginning to the `qsym2-output` logger.
pub(crate) fn log_macsec_begin(sectitle: &str) {
    qsym2_output!("<<<<< [Begin] {sectitle}");
}

/// Logs a nicely formatted macro-section ending to the `qsym2-output` logger.
pub(crate) fn log_macsec_end(sectitle: &str) {
    qsym2_output!(">>>>> [End] {sectitle}");
}

/// Turns a boolean into a string of `yes` or `no`.
pub(crate) fn nice_bool(b: bool) -> String {
    if b {
        "yes".to_string()
    } else {
        "no".to_string()
    }
}

/// A trait for logging `QSym2` outputs nicely.
pub(crate) trait QSym2Output: fmt::Debug + fmt::Display {
    /// Logs display output nicely.
    fn log_output_display(&self) {
        let lines = self.to_string();
        lines.lines().for_each(|line| {
            qsym2_output!("{line}");
        })
    }

    /// Logs debug output nicely.
    fn log_output_debug(&self) {
        let lines = format!("{self:?}");
        lines.lines().for_each(|line| {
            qsym2_output!("{line}");
        })
    }
}

// Blanket implementation
impl<T> QSym2Output for T where T: fmt::Debug + fmt::Display {}
