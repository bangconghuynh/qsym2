use crate::group::GroupProperties;
use crate::chartab::chartab_symbols::ReducibleLinearSpaceSymbol;

trait IntoOrbit<G> where
    G: GroupProperties,
    Self: Sized,
{
    type OrbitIntoIter: Clone + IntoIterator<Item = Self>;

    fn orbit_iter(&self, group: &G) -> Self::OrbitIntoIter;
}

trait RepAnalysisResult<G, R> where
    G: GroupProperties,
    R: ReducibleLinearSpaceSymbol
{
    fn actual_group(&self) -> &G;

    fn finite_group(&self) -> &G;

    fn main_rep(&self) -> &R;
}

trait RepAnalysis<G, R> where
    G: GroupProperties,
    R: ReducibleLinearSpaceSymbol
{
}
