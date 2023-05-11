use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

use approx;
use nalgebra::{Matrix3, Point3, Rotation3, Translation3, UnitVector3, Vector3};
use num_traits::ToPrimitive;
use periodic_table;

use crate::aux::geometry::{self, ImproperRotationKind, Transform};
use crate::aux::misc::{self, HashableFloat};

/// A struct storing a look-up of element symbols to give atomic numbers
/// and atomic masses.
pub struct ElementMap<'a> {
    /// A [HashMap] from a symbol string to a tuple of atomic number and atomic
    /// mass.
    pub map: HashMap<&'a str, (u32, f64)>,
}

impl Default for ElementMap<'static> {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementMap<'static> {
    /// Creates a new [`ElementMap`] for all elements in the periodic table.
    #[must_use]
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
/// The numeric mass value.
fn parse_atomic_mass(mass_str: &str) -> f64 {
    let mass = mass_str.replace(&['(', ')', '[', ']'][..], "");
    mass.parse::<f64>()
        .unwrap_or_else(|_| panic!("Unable to parse atomic mass string {mass}."))
}

/// A struct representing an atom.
#[derive(Clone)]
pub struct Atom {
    /// The atom kind.
    pub kind: AtomKind,

    /// The atomic number of the atom.
    pub atomic_number: u32,

    /// The atomic symbol of the atom.
    pub atomic_symbol: String,

    /// The weighted-average atomic mass for all naturally occuring isotopes.
    pub atomic_mass: f64,

    /// The position of the atom.
    pub coordinates: Point3<f64>,

    /// A threshold for approximate equality comparisons.
    pub threshold: f64,
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
    /// * `thresh` - A threshold for approximate equality comparisons.
    ///
    /// # Returns
    ///
    /// The parsed [`Atom`] struct if the line has the correct format,
    /// otherwise [`None`].
    #[must_use]
    pub fn from_xyz(line: &str, emap: &ElementMap, thresh: f64) -> Option<Atom> {
        let split: Vec<&str> = line.split_whitespace().collect();
        if split.len() != 4 {
            return None;
        };
        let atomic_symbol = split.first().expect("Unable to get the element symbol.");
        let (atomic_number, atomic_mass) = emap
            .map
            .get(atomic_symbol)
            .expect("Invalid atomic symbol encountered.");
        let coordinates = Point3::new(
            split
                .get(1)
                .expect("Unable to get the x coordinate.")
                .parse::<f64>()
                .expect("Unable to parse the x coordinate."),
            split
                .get(2)
                .expect("Unable to get the y coordinate.")
                .parse::<f64>()
                .expect("Unable to parse the y coordinate."),
            split
                .get(3)
                .expect("Unable to get the z coordinate.")
                .parse::<f64>()
                .expect("Unable to parse the z coordinate."),
        );
        let atom = Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *atomic_number,
            atomic_symbol: (*atomic_symbol).to_string(),
            atomic_mass: *atomic_mass,
            coordinates,
            threshold: thresh,
        };
        Some(atom)
    }

    /// Creates an ordinary atom.
    ///
    /// Arguments
    ///
    /// * coordinates - The coordinates of the special atom.
    ///
    /// Rerturns
    ///
    /// The required ordinary atom.
    #[must_use]
    pub fn new_ordinary(
        atomic_symbol: &str,
        coordinates: Point3<f64>,
        emap: &ElementMap,
        thresh: f64,
    ) -> Atom {
        let (atomic_number, atomic_mass) = emap
            .map
            .get(atomic_symbol)
            .expect("Invalid atomic symbol encountered.");
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *atomic_number,
            atomic_symbol: atomic_symbol.to_string(),
            atomic_mass: *atomic_mass,
            coordinates,
            threshold: thresh,
        }
    }

    /// Creates a special atom.
    ///
    /// Arguments
    ///
    /// * kind - The required special kind.
    /// * coordinates - The coordinates of the special atom.
    ///
    /// Rerturns
    ///
    /// `None` if `kind` is not one of the special atom kinds, `Some<Atom>`
    /// otherwise.
    #[must_use]
    pub fn new_special(kind: AtomKind, coordinates: Point3<f64>, thresh: f64) -> Option<Atom> {
        match kind {
            AtomKind::Magnetic(_) | AtomKind::Electric(_) => Some(Atom {
                kind,
                atomic_number: 0,
                atomic_symbol: String::new(),
                atomic_mass: 100.0,
                coordinates,
                threshold: thresh,
            }),
            AtomKind::Ordinary => None,
        }
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = self
            .threshold
            .log10()
            .abs()
            .round()
            .to_usize()
            .ok_or_else(|| fmt::Error)?
            + 1;
        let length = (precision + precision.div_euclid(2)).max(6);
        if let AtomKind::Ordinary = self.kind {
            write!(
                f,
                "{:>9} {:>3} {:+length$.precision$} {:+length$.precision$} {:+length$.precision$}",
                "Atom",
                self.atomic_symbol,
                self.coordinates[0],
                self.coordinates[1],
                self.coordinates[2],
            )
        } else {
            write!(
                f,
                "{:>13} {:+length$.precision$} {:+length$.precision$} {:+length$.precision$}",
                self.kind,
                self.coordinates[0],
                self.coordinates[1],
                self.coordinates[2],
            )
        }
    }
}

impl fmt::Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

/// An enum describing the atom kind.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum AtomKind {
    /// An ordinary atom.
    Ordinary,

    /// A fictitious atom representing a magnetic field. This variant contains
    /// a flag to indicate if the fictitious atom is of positive type or not.
    Magnetic(bool),

    /// A fictitious atom representing an electric field. This variant contains
    /// a flag to indicate if the fictitious atom is of positive type or not.
    Electric(bool),
}

impl fmt::Display for AtomKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ordinary => write!(f, "{}", "Atom".to_owned()),
            Self::Magnetic(pos) => {
                if *pos {
                    write!(f, "{}", "MagneticAtom+".to_owned())
                } else {
                    write!(f, "{}", "MagneticAtom-".to_owned())
                }
            }
            Self::Electric(pos) => {
                if *pos {
                    write!(f, "{}", "ElectricAtom+".to_owned())
                } else {
                    write!(f, "{}", "ElectricAtom-".to_owned())
                }
            }
        }
    }
}

impl Transform for Atom {
    fn transform_mut(&mut self, mat: &Matrix3<f64>) {
        let det = mat.determinant();
        assert!(
            approx::relative_eq!(
                det,
                1.0,
                epsilon = self.threshold,
                max_relative = self.threshold
            ) || approx::relative_eq!(
                det,
                -1.0,
                epsilon = self.threshold,
                max_relative = self.threshold
            )
        );
        self.coordinates = mat * self.coordinates;
        if approx::relative_eq!(
            det,
            -1.0,
            epsilon = self.threshold,
            max_relative = self.threshold
        ) {
            if let AtomKind::Magnetic(pos) = self.kind {
                self.kind = AtomKind::Magnetic(!pos);
            }
        };
    }

    fn rotate_mut(&mut self, angle: f64, axis: &Vector3<f64>) {
        let normalised_axis = UnitVector3::new_normalize(*axis);
        let rotation = Rotation3::from_axis_angle(&normalised_axis, angle);
        self.coordinates = rotation.transform_point(&self.coordinates);
    }

    fn improper_rotate_mut(
        &mut self,
        angle: f64,
        axis: &Vector3<f64>,
        kind: &ImproperRotationKind,
    ) {
        let mat = geometry::improper_rotation_matrix(angle, axis, 1, kind);
        self.transform_mut(&mat);
    }

    fn translate_mut(&mut self, tvec: &Vector3<f64>) {
        let translation = Translation3::from(*tvec);
        self.coordinates = translation.transform_point(&self.coordinates);
    }

    fn recentre_mut(&mut self) {
        self.coordinates = Point3::origin();
    }

    fn reverse_time_mut(&mut self) {
        if let AtomKind::Magnetic(polarity) = self.kind {
            self.kind = AtomKind::Magnetic(!polarity);
        }
    }

    fn transform(&self, mat: &Matrix3<f64>) -> Self {
        let mut transformed_atom = self.clone();
        transformed_atom.transform_mut(mat);
        transformed_atom
    }

    fn rotate(&self, angle: f64, axis: &Vector3<f64>) -> Self {
        let mut rotated_atom = self.clone();
        rotated_atom.rotate_mut(angle, axis);
        rotated_atom
    }

    fn improper_rotate(
        &self,
        angle: f64,
        axis: &Vector3<f64>,
        kind: &ImproperRotationKind,
    ) -> Self {
        let mut improper_rotated_atom = self.clone();
        improper_rotated_atom.improper_rotate_mut(angle, axis, kind);
        improper_rotated_atom
    }

    fn translate(&self, tvec: &Vector3<f64>) -> Self {
        let mut translated_atom = self.clone();
        translated_atom.translate_mut(tvec);
        translated_atom
    }

    fn recentre(&self) -> Self {
        let mut recentred_atom = self.clone();
        recentred_atom.recentre_mut();
        recentred_atom
    }

    fn reverse_time(&self) -> Self {
        let mut time_reversed_atom = self.clone();
        time_reversed_atom.reverse_time_mut();
        time_reversed_atom
    }
}

impl PartialEq for Atom {
    /// The `[Self::threshold]` value for each atom defines a discrete grid over
    /// the real number field. All real numbers (*e.g.* atomic mass, coordinates)
    /// are rounded to take on the values on this discrete grid which are then
    /// used for hashing and comparisons.
    fn eq(&self, other: &Self) -> bool {
        let result = self.atomic_number == other.atomic_number
            && self.kind == other.kind
            && approx::relative_eq!(
                self.atomic_mass.round_factor(self.threshold),
                other.atomic_mass.round_factor(other.threshold),
            )
            && approx::relative_eq!(
                self.coordinates[0].round_factor(self.threshold),
                other.coordinates[0].round_factor(other.threshold),
            )
            && approx::relative_eq!(
                self.coordinates[1].round_factor(self.threshold),
                other.coordinates[1].round_factor(other.threshold),
            )
            && approx::relative_eq!(
                self.coordinates[2].round_factor(self.threshold),
                other.coordinates[2].round_factor(other.threshold),
            );
        if result {
            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
        }
        result
    }
}

impl Eq for Atom {}

impl Hash for Atom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.atomic_number.hash(state);
        self.kind.hash(state);
        self.atomic_mass
            .round_factor(self.threshold)
            .integer_decode()
            .hash(state);
        self.coordinates[0]
            .round_factor(self.threshold)
            .integer_decode()
            .hash(state);
        self.coordinates[1]
            .round_factor(self.threshold)
            .integer_decode()
            .hash(state);
        self.coordinates[2]
            .round_factor(self.threshold)
            .integer_decode()
            .hash(state);
    }
}
