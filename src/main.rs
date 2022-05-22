use clap::{app_from_crate, arg};
use std::process;

use rustyinspect::aux::molecule::Molecule;
use rustyinspect::rotsym;

fn main() {
    let matches = app_from_crate!()
        .arg(arg!([XYZ_FILE] "xyz file"))
        .arg(
            arg!(-t --threshold <THRESHOLD> "Threshold for moment of inertia comparison")
                .required(false)
                .default_value("1e-6"),
        )
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
    let verbose = matches.occurrences_of("verbose");

    let mol = Molecule::from_xyz(filename, 1e-4);
    let com = mol.calc_com(verbose);
    let inertia = mol.calc_inertia_tensor(&com, verbose);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, thresh, verbose);
    println!("Rotational symmetry: {}", rotsym_result);
    let sea_groups = mol.calc_sea_groups(1);
    println!("SEAs: {:?}", sea_groups);
}
