use clap::{app_from_crate, arg};
use std::process;

use rustyinspect::aux::molecule::Molecule;
use rustyinspect::rotsym::calc;

fn main() {
    let matches = app_from_crate!()
        .arg(arg!([XYZ_FILE] "xyz file"))
        .arg(
            arg!(-t --threshold <THRESHOLD> "Threshold for moment of inertia comparison")
                .required(false)
                .default_value("1e-6"),
        )
        .arg(arg!(-a --abs_compare "Absolute moment of inertia comparison"))
        .arg(arg!(-v --verbose ... "Use verbose output. Maybe specified twice for 'very verbose'."))
        .get_matches();

    let filename = matches.value_of("XYZ_FILE").unwrap_or_else(|| {
        println!("No xyz file provided.");
        process::exit(1);
    });
    let thresh = matches
        .value_of("threshold")
        .unwrap()
        .parse::<f64>()
        .unwrap();
    let abs_compare = matches.is_present("abs_compare");
    let verbose = matches.occurrences_of("verbose");

    let mol = Molecule::from_xyz(filename);
    let com = mol.calc_com(verbose);
    let inertia = mol.calc_moi(&com, verbose);
    let rotsym_result = calc::calc_rotational_symmetry(&inertia, thresh, verbose, abs_compare);
    println!("Rotational symmetry: {}", rotsym_result);
}
