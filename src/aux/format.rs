use std::fmt;

use log;

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
    log::info!(target: "qsym2-output", "┌──{}──┐", bar);
    log::info!(target: "qsym2-output", "│§ {} §│", title);
    log::info!(target: "qsym2-output", "└──{}──┘", bar);
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
    log::info!(target: "qsym2-output", "{}", subtitle);
    log::info!(target: "qsym2-output", "{}", bar);
}

/// Logs a nicely formatted macro-section beginning to the `qsym2-output` logger.
pub(crate) fn log_macsec_begin(sectitle: &str) {
    log::info!(target: "qsym2-output", "<<<<< [Begin] {sectitle}");
}

/// Logs a nicely formatted macro-section ending to the `qsym2-output` logger.
pub(crate) fn log_macsec_end(sectitle: &str) {
    log::info!(target: "qsym2-output", ">>>>> [End] {sectitle}");
}

/// Turns a boolean into a string of `yes` or `no`.
pub(crate) fn nice_bool(b: bool) -> String {
    if b { "yes".to_string() } else { "no".to_string() }
}

/// A trait for logging `QSym2` outputs nicely.
pub(crate) trait QSym2Output: fmt::Debug + fmt::Display {
    /// Logs display output nicely.
    fn log_output_display(&self) {
        let lines = self.to_string();
        lines.lines().for_each(|line| {
            log::info!(target: "qsym2-output", "{line}");
        })
    }

    /// Logs debug output nicely.
    fn log_output_debug(&self) {
        let lines = format!("{self:?}");
        lines.lines().for_each(|line| {
            log::info!(target: "qsym2-output", "{line}");
        })
    }
}

// Blanket implementation
impl<T> QSym2Output for T where T: fmt::Debug + fmt::Display {}
