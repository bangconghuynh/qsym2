use clap::{app_from_crate, arg};
use std::process;

use qsym2::aux::molecule::Molecule;
use qsym2::rotsym;

fn main() {
    let matches = app_from_crate!()
        .arg(arg!([XYZ_FILE] "xyz file"))
        .arg(
            arg!(-t --threshold <THRESHOLD> "Threshold for moment of inertia comparison")
                .required(false)
                .default_value("1e-6"),
        )
        .get_matches();

    let filename = matches.value_of("XYZ_FILE").unwrap_or_else(|| {
        println!("No xyz file provided.");
        process::exit(1);
    });
    let thresh = matches
        .value_of("threshold")
        .expect("Threshold value not found.")
        .parse::<f64>()
        .expect("Unable to parse threshold value.");

    let mol = Molecule::from_xyz(filename, 1e-4);
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, thresh);
    println!("Rotational symmetry: {rotsym_result}");
    let sea_groups = mol.calc_sea_groups();
    println!("SEAs: {sea_groups:?}");
}
