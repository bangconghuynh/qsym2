use crate::aux::atom::{Atom, AtomKind, ElementMap};
use crate::aux::molecule::Molecule;
use nalgebra::Point3;


pub fn gen_twisted_h8(theta: f64) -> Molecule {
    let emap = ElementMap::new();
    let (atomic_number, atomic_mass) = emap.map.get("H").unwrap();
    let atoms: [Atom; 8] = [
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *atomic_number,
            atomic_symbol: "H".to_owned(),
            atomic_mass: *atomic_mass,
            coordinates: Point3::new(0.0, 0.0, 0.0),
            threshold: 1e-7
        },
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *atomic_number,
            atomic_symbol: "H".to_owned(),
            atomic_mass: *atomic_mass,
            coordinates: Point3::new(1.0, 1.0, 0.0),
            threshold: 1e-7
        },
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *atomic_number,
            atomic_symbol: "H".to_owned(),
            atomic_mass: *atomic_mass,
            coordinates: Point3::new(1.0, 0.0, 0.0),
            threshold: 1e-7
        },
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *atomic_number,
            atomic_symbol: "H".to_owned(),
            atomic_mass: *atomic_mass,
            coordinates: Point3::new(0.0, 1.0, 0.0),
            threshold: 1e-7
        },
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *atomic_number,
            atomic_symbol: "H".to_owned(),
            atomic_mass: *atomic_mass,
            coordinates: Point3::new(
                1.0/(2.0_f64.sqrt()) * (5.0 * std::f64::consts::FRAC_PI_4 + theta).cos() + 0.5,
                1.0/(2.0_f64.sqrt()) * (5.0 * std::f64::consts::FRAC_PI_4 + theta).sin() + 0.5,
                2.0
            ),
            threshold: 1e-7
        },
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *atomic_number,
            atomic_symbol: "H".to_owned(),
            atomic_mass: *atomic_mass,
            coordinates: Point3::new(
                1.0/(2.0_f64.sqrt()) * (1.0 * std::f64::consts::FRAC_PI_4 + theta).cos() + 0.5,
                1.0/(2.0_f64.sqrt()) * (1.0 * std::f64::consts::FRAC_PI_4 + theta).sin() + 0.5,
                2.0
            ),
            threshold: 1e-7
        },
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *atomic_number,
            atomic_symbol: "H".to_owned(),
            atomic_mass: *atomic_mass,
            coordinates: Point3::new(
                1.0/(2.0_f64.sqrt()) * (7.0 * std::f64::consts::FRAC_PI_4 + theta).cos() + 0.5,
                1.0/(2.0_f64.sqrt()) * (7.0 * std::f64::consts::FRAC_PI_4 + theta).sin() + 0.5,
                2.0
            ),
            threshold: 1e-7
        },
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *atomic_number,
            atomic_symbol: "H".to_owned(),
            atomic_mass: *atomic_mass,
            coordinates: Point3::new(
                1.0/(2.0_f64.sqrt()) * (3.0 * std::f64::consts::FRAC_PI_4 + theta).cos() + 0.5,
                1.0/(2.0_f64.sqrt()) * (3.0 * std::f64::consts::FRAC_PI_4 + theta).sin() + 0.5,
                2.0
            ),
            threshold: 1e-7
        },
    ];
    Molecule::from_atoms(&atoms, 1e-7)
}


pub fn gen_arbitrary_half_sandwich(n: u32) -> Molecule {
    let emap = ElementMap::new();
    let mut atoms: Vec<Atom> = vec![];
    let (h_atomic_number, h_atomic_mass) = emap.map.get("H").unwrap();
    let (c_atomic_number, c_atomic_mass) = emap.map.get("C").unwrap();
    let (v_atomic_number, v_atomic_mass) = emap.map.get("V").unwrap();
    atoms.push(
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *v_atomic_number,
            atomic_symbol: "V".to_owned(),
            atomic_mass: *v_atomic_mass,
            coordinates: Point3::new(0.0, 0.0, 1.0 + 0.1 * (n as f64)),
            threshold: 1e-7
        },
    );
    for i in 0..n {
        atoms.push(
            Atom {
                kind: AtomKind::Ordinary,
                atomic_number: *c_atomic_number,
                atomic_symbol: "C".to_owned(),
                atomic_mass: *c_atomic_mass,
                coordinates: Point3::new(
                    1.0 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).cos(),
                    1.0 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).sin(),
                    0.0),
                threshold: 1e-7
            }
        );
        atoms.push(
            Atom {
                kind: AtomKind::Ordinary,
                atomic_number: *h_atomic_number,
                atomic_symbol: "H".to_owned(),
                atomic_mass: *h_atomic_mass,
                coordinates: Point3::new(
                    1.5 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).cos(),
                    1.5 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).sin(),
                    0.0),
                threshold: 1e-7
            }
        );
    };
    Molecule::from_atoms(&atoms, 1e-7)
}

pub fn gen_arbitrary_eclipsed_sandwich(n: u32) -> Molecule {
    let emap = ElementMap::new();
    let mut atoms: Vec<Atom> = vec![];
    let (h_atomic_number, h_atomic_mass) = emap.map.get("H").unwrap();
    let (c_atomic_number, c_atomic_mass) = emap.map.get("C").unwrap();
    let (m_atomic_number, m_atomic_mass) = emap.map.get("Co").unwrap();
    atoms.push(
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *m_atomic_number,
            atomic_symbol: "Co".to_owned(),
            atomic_mass: *m_atomic_mass,
            coordinates: Point3::new(0.0, 0.0, 0.0),
            threshold: 1e-7
        },
    );
    for i in 0..n {
        atoms.push(
            Atom {
                kind: AtomKind::Ordinary,
                atomic_number: *c_atomic_number,
                atomic_symbol: "C".to_owned(),
                atomic_mass: *c_atomic_mass,
                coordinates: Point3::new(
                    1.0 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).cos(),
                    1.0 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).sin(),
                    -1.0 - 0.1 * (n as f64)),
                threshold: 1e-7
            }
        );
        atoms.push(
            Atom {
                kind: AtomKind::Ordinary,
                atomic_number: *h_atomic_number,
                atomic_symbol: "H".to_owned(),
                atomic_mass: *h_atomic_mass,
                coordinates: Point3::new(
                    1.5 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).cos(),
                    1.5 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).sin(),
                    -1.0 - 0.1 * (n as f64)),
                threshold: 1e-7
            }
        );
        atoms.push(
            Atom {
                kind: AtomKind::Ordinary,
                atomic_number: *c_atomic_number,
                atomic_symbol: "C".to_owned(),
                atomic_mass: *c_atomic_mass,
                coordinates: Point3::new(
                    1.0 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).cos(),
                    1.0 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).sin(),
                    1.0 + 0.1 * (n as f64)),
                threshold: 1e-7
            }
        );
        atoms.push(
            Atom {
                kind: AtomKind::Ordinary,
                atomic_number: *h_atomic_number,
                atomic_symbol: "H".to_owned(),
                atomic_mass: *h_atomic_mass,
                coordinates: Point3::new(
                    1.5 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).cos(),
                    1.5 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).sin(),
                    1.0 + 0.1 * (n as f64)),
                threshold: 1e-7
            }
        );
    };
    Molecule::from_atoms(&atoms, 1e-7)
}

pub fn gen_arbitrary_twisted_sandwich(n: u32, frac: f64) -> Molecule {
    let emap = ElementMap::new();
    let mut atoms: Vec<Atom> = vec![];
    let (h_atomic_number, h_atomic_mass) = emap.map.get("H").unwrap();
    let (c_atomic_number, c_atomic_mass) = emap.map.get("C").unwrap();
    let (m_atomic_number, m_atomic_mass) = emap.map.get("Co").unwrap();
    atoms.push(
        Atom {
            kind: AtomKind::Ordinary,
            atomic_number: *m_atomic_number,
            atomic_symbol: "Co".to_owned(),
            atomic_mass: *m_atomic_mass,
            coordinates: Point3::new(0.0, 0.0, 0.0),
            threshold: 1e-7
        },
    );
    for i in 0..n {
        atoms.push(
            Atom {
                kind: AtomKind::Ordinary,
                atomic_number: *c_atomic_number,
                atomic_symbol: "C".to_owned(),
                atomic_mass: *c_atomic_mass,
                coordinates: Point3::new(
                    1.0 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).cos(),
                    1.0 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).sin(),
                    -1.0 - 0.1 * (n as f64)),
                threshold: 1e-7
            }
        );
        atoms.push(
            Atom {
                kind: AtomKind::Ordinary,
                atomic_number: *h_atomic_number,
                atomic_symbol: "H".to_owned(),
                atomic_mass: *h_atomic_mass,
                coordinates: Point3::new(
                    1.5 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).cos(),
                    1.5 * ((i as f64) * 2.0 * std::f64::consts::PI / (n as f64)).sin(),
                    -1.0 - 0.1 * (n as f64)),
                threshold: 1e-7
            }
        );
        atoms.push(
            Atom {
                kind: AtomKind::Ordinary,
                atomic_number: *c_atomic_number,
                atomic_symbol: "C".to_owned(),
                atomic_mass: *c_atomic_mass,
                coordinates: Point3::new(
                    1.0 * ((i as f64 + frac) * 2.0 * std::f64::consts::PI / (n as f64)).cos(),
                    1.0 * ((i as f64 + frac) * 2.0 * std::f64::consts::PI / (n as f64)).sin(),
                    1.0 + 0.1 * (n as f64)),
                threshold: 1e-7
            }
        );
        atoms.push(
            Atom {
                kind: AtomKind::Ordinary,
                atomic_number: *h_atomic_number,
                atomic_symbol: "H".to_owned(),
                atomic_mass: *h_atomic_mass,
                coordinates: Point3::new(
                    1.5 * ((i as f64 + frac) * 2.0 * std::f64::consts::PI / (n as f64)).cos(),
                    1.5 * ((i as f64 + frac) * 2.0 * std::f64::consts::PI / (n as f64)).sin(),
                    1.0 + 0.1 * (n as f64)),
                threshold: 1e-7
            }
        );
    };
    Molecule::from_atoms(&atoms, 1e-7)
}
