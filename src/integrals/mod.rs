mod shell_tuple;

macro_rules! count_exprs {
    () => (0);
    ($head:expr) => (1);
    ($head:expr, $($tail:expr),*) => (1 + count_exprs!($($tail),*));
}

macro_rules! replace_expr {
    ($_t:tt $sub:expr) => {$sub};
}

use {count_exprs, replace_expr};
