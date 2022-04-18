use nalgebra::Point3;
use periodic_table;
use std::collections::HashMap;

/// A struct storing a look-up of element symbols to give atomic numbers
/// and atomic masses.
pub struct ElementMap<'a> {
    /// A [HashMap] from a symbol string to a tuple of atomic number and atomic
    /// mass.
    map: HashMap<&'a str, (u32, f64)>,
}

impl ElementMap<'static> {
    /// Creates a new [`ElementMap`] for all elements in the periodic table.
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

/// An auxiliary function that parses the atomic mass string in the format of
/// [`periodic_table`] to a single float value.
///
/// # Arguments
///
/// * `mass_str` - A string of mass value that is either `x.y(z)` where the
///     uncertain digit `z` is enclosed in parentheses, or `[x]` where `x`
///     is the mass number in place of precise experimental values.
///
/// # Returns
///
/// The mass value as a float.
fn parse_atomic_mass(mass_str: &str) -> f64 {
    let mass = mass_str.replace(&['(', ')', '[', ']'][..], "");
    mass.parse::<f64>().unwrap()
}

/// A struct representing an atom.
#[derive(Debug)]
pub struct Atom {
    /// The atomic number of the atom.
    atomic_number: u32,
    /// The atomic symbol of the atom.
    atomic_symbol: String,
    /// The weighted-average atomic mass for all naturally occuring isotopes.
    pub atomic_mass: f64,
    /// The position of the atom.
    pub coordinates: Point3<f64>,
}

impl Atom {
    /// Parses an atom line in an `xyz` file to construct an [`Atom`].
    ///
    /// # Arguments
    ///
    /// * `line` - A line in an `xyz` file containing an atomic symbol and
    ///     three Cartesian coordinates.
    /// * `emap` - A hash map between atomic symbols and atomic numbers and
    ///     masses.
    ///
    /// # Returns
    ///
    /// The parsed [`Atom`] struct if the line has the correct format,
    /// otherwise [`None`].
    pub fn from_xyz(line: &str, emap: &ElementMap) -> Option<Atom> {
        let split: Vec<&str> = line.split_whitespace().collect();
        if split.len() != 4 {
            return None;
        };
        let atomic_symbol = split.get(0).unwrap();
        let (atomic_number, atomic_mass) = emap
            .map
            .get(atomic_symbol)
            .expect("Invalid atomic symbol encountered.");
        let coordinates = Point3::new(
            split.get(1).unwrap().parse::<f64>().unwrap(),
            split.get(2).unwrap().parse::<f64>().unwrap(),
            split.get(3).unwrap().parse::<f64>().unwrap(),
        );
        let atom = Atom {
            atomic_number: *atomic_number,
            atomic_symbol: atomic_symbol.to_string(),
            atomic_mass: *atomic_mass,
            coordinates,
        };
        Some(atom)
    }
}
