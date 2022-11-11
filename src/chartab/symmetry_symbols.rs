use std::hash::Hash;

trait MathematicalSymbol: Hash {
    /// The main part of the symbol.
    fn main(&self) -> String;

    /// The pre-superscript part of the symbol.
    fn presuper(&self) -> String;

    /// The pre-subscript part of the symbol.
    fn presub(&self) -> String;

    /// The post-superscript part of the symbol.
    fn postsuper(&self) -> String;

    /// The post-subscript part of the symbol.
    fn postsub(&self) -> String;

    /// The prefactor part of the symbol.
    fn prefactor(&self) -> String;

    /// The postfactor part of the symbol.
    fn postfactor(&self) -> String;

    /// The postfactor part of the symbol.
    fn postfactor(&self) -> String;

    /// The LaTeX string for this symbol.
    fn latex_fmt(&self) -> String;
}
