use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::aux::misc::HashableFloat;


/// An enum to handle symmetry element orders which can be integers, floats, or infinity.
#[derive(Clone, Debug)]
pub enum ElementOrder {
    /// Positive integer order.
    Int(u32),

    /// Infinite order.
    Inf,
}

impl ElementOrder {
    pub fn new(order: f64, thresh: f64) -> Self {
        assert!(
            order.is_sign_positive(),
            "Order value {} is invalid. Order values must be strictly positive.",
            order
        );
        if order.is_infinite() {
            return Self::Inf;
        }
        let rounded_order = order.round_factor(thresh);
        if approx::relative_eq!(
            rounded_order,
            rounded_order.round(),
            epsilon = thresh,
            max_relative = thresh
        ) {
            return Self::Int(rounded_order as u32);
        }
        panic!("The input order is not an integer.");
    }

    pub fn to_float(&self) -> f64 {
        match self {
            Self::Int(s_i) => *s_i as f64,
            Self::Inf => f64::INFINITY,
        }
    }
}

impl PartialEq for ElementOrder {
    fn eq(&self, other: &Self) -> bool {
        match &self {
            Self::Int(s_i) => match &other {
                Self::Int(o_i) => {
                    return s_i == o_i;
                }
                Self::Inf => return false,
            },
            Self::Inf => match &other {
                Self::Int(_) => return false,
                Self::Inf => return true,
            },
        }
    }
}

impl Eq for ElementOrder {}

impl PartialOrd for ElementOrder {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.to_float().partial_cmp(&other.to_float())?)
    }
}

impl Ord for ElementOrder {
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_float().total_cmp(&other.to_float())
    }
}

impl Hash for ElementOrder {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &self {
            Self::Int(s_i) => {
                s_i.hash(state);
            }
            Self::Inf => {
                f64::INFINITY.integer_decode().hash(state);
            }
        }
    }
}

impl fmt::Display for ElementOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            Self::Int(s_i) => write!(f, "{}", s_i),
            Self::Inf => write!(f, "{}", "∞".to_owned()),
        }
    }
}

pub const ORDER_1: ElementOrder = ElementOrder::Int(1);
pub const ORDER_2: ElementOrder = ElementOrder::Int(2);
pub const ORDER_I: ElementOrder = ElementOrder::Inf;
