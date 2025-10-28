//! Symmetry projection via group orbits.

use anyhow::format_err;

use crate::analysis::Orbit;
use crate::chartab::CharacterTable;
use crate::chartab::character::Character;
use crate::chartab::chartab_group::CharacterProperties;
use crate::group::GroupProperties;

/// Trait to facilitate the application of a group projection operator.
pub trait Projectable<G, I>: Orbit<G, I>
where
    G: GroupProperties + CharacterProperties,
{
    /// The type of the result of the projection.
    type Projected<'p>
    where
        Self: 'p;

    // ----------------
    // Required methods
    // ----------------
    /// Projects the orbit onto a symmetry subspace.
    fn project_onto(&self, row: &G::RowSymbol) -> Self::Projected<'_>;

    // ----------------
    // Provided methods
    // ----------------
    /// Returns an iterator containing each term in the projection summation and the accompanying
    /// character value (without complex conjugation).
    fn generate_orbit_algebra_terms<'a>(
        &'a self,
        row: &G::RowSymbol,
    ) -> impl Iterator<Item = Result<(&'a Character, I), anyhow::Error>>
    where
        I: 'a,
        G: 'a,
    {
        self.iter().enumerate().map(|(i, item_res)| {
            let chr = self
                .group()
                .get_cc_of_element_index(i)
                .and_then(|cc_i| self.group().get_cc_symbol_of_index(cc_i))
                .map(|cc| self.group().character_table().get_character(row, &cc))
                .ok_or_else(|| {
                    format_err!(
                        "Unable to obtain the character of the row {} for element index {i}.",
                        row.clone()
                    )
                })?;
            item_res.map(|item| (chr, item))
        })
    }
}
