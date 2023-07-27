macro_rules! count_exprs {
    () => (0);
    ($head:expr) => (1);
    ($head:expr, $($tail:expr),*) => (1 + count_exprs!($($tail),*));
}

macro_rules! replace_expr {
    ($_t:tt $sub:expr) => {$sub};
}

pub mod analysis;
pub mod angmom;
pub mod auxiliary;
pub mod basis;
pub mod bindings;
pub mod chartab;
pub mod drivers;
pub mod group;
#[cfg(feature = "integrals")]
pub mod integrals;
pub mod interfaces;
pub mod io;
pub mod permutation;
pub mod rotsym;
pub mod symmetry;
pub mod target;
