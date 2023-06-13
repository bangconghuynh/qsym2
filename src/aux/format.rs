use std::fmt;

use log;

pub(crate) fn write_title(f: &mut fmt::Formatter<'_>, title: &str) -> fmt::Result {
    let length = title.chars().count();
    let bar = "─".repeat(length);
    writeln!(f, "┌──{bar}──┐")?;
    writeln!(f, "│§ {title} §│")?;
    writeln!(f, "└──{bar}──┘")?;
    Ok(())
}

pub(crate) fn log_title(title: &str) {
    let length = title.chars().count();
    let bar = "─".repeat(length);
    log::info!(target: "qsym2-output", "┌──{}──┐", bar);
    log::info!(target: "qsym2-output", "│§ {} §│", title);
    log::info!(target: "qsym2-output", "└──{}──┘", bar);
}

pub(crate) fn write_subtitle(f: &mut fmt::Formatter<'_>, subtitle: &str) -> fmt::Result {
    let length = subtitle.chars().count();
    let bar = "═".repeat(length);
    writeln!(f, "{subtitle}")?;
    writeln!(f, "{bar}")?;
    Ok(())
}

pub(crate) fn log_subtitle(subtitle: &str) {
    let length = subtitle.chars().count();
    let bar = "═".repeat(length);
    log::info!(target: "qsym2-output", "{}", subtitle);
    log::info!(target: "qsym2-output", "{}", bar);
}

pub(crate) fn nice_bool(b: bool) -> String {
    if b { "yes".to_string() } else { "no".to_string() }
}
