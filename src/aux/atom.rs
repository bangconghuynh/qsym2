use periodic_table;
use std::collections::HashMap;
use std::fs;
use std::process;

use crate::aux::geometry::Point3D;


pub struct ElementMap<'a> {
    map: HashMap<&'a str, (u32, f64)>,
}

fn parse_atomic_mass(mass_str: &str) -> f64 {
    let mass = mass_str.replace(&['(', ')', '[', ']'][..], "");
    mass.parse::<f64>().unwrap()
}

impl ElementMap<'static> {
    pub fn new() -> ElementMap<'static> {
        let mut map = HashMap::new();
        let elements = periodic_table::periodic_table();
        for element in elements {
            let mass = parse_atomic_mass(element.atomic_mass);
            map.insert(element.symbol, (element.atomic_number, mass));
        }
        ElementMap { map }
    }
}

#[derive(Debug)]
pub struct Atom {
    atomic_number: u32,
    atomic_symbol: String,
    pub atomic_mass: f64,
    pub coordinates: Point3D<f64>,
}

impl Atom {
    pub fn from_xyz(line: &str, emap: &ElementMap) -> Option<Atom> {
        let split: Vec<&str> = line.split_whitespace().collect();
        if split.len() != 4 {
            return None;
        };
        let atomic_symbol = split.get(0).unwrap();
        let (atomic_number, atomic_mass) = emap.map.get(atomic_symbol).expect("Invalid atomic symbol encountered.");
        let coordinates = Point3D {
            x: split.get(1).unwrap().parse::<f64>().unwrap(),
            y: split.get(2).unwrap().parse::<f64>().unwrap(),
            z: split.get(3).unwrap().parse::<f64>().unwrap(),
        };
        let atom = Atom {
            atomic_number: *atomic_number,
            atomic_symbol: atomic_symbol.to_string(),
            atomic_mass: *atomic_mass,
            coordinates,
        };
        Some(atom)
    }
}

pub fn parse_xyz(filename: &str) -> Vec<Atom> {
    let contents = fs::read_to_string(filename).unwrap_or_else(|err| {
        println!("Unable to read file {}.", filename);
        println!("{}", err);
        process::exit(1);
    });

    let mut atoms: Vec<Atom> = vec![];
    let emap = ElementMap::new();
    let mut n_atoms = 0usize;
    for (i, line) in contents.lines().enumerate() {
        if i == 0 {
            n_atoms = line.parse::<usize>().unwrap();
        } else if i == 1 {
            continue;
        } else {
            atoms.push(Atom::from_xyz(&line, &emap).unwrap());
        }
    }
    assert_eq!(
        atoms.len(),
        n_atoms,
        "Expected {} atoms, got {} instead.",
        n_atoms,
        atoms.len()
    );
    atoms
}
