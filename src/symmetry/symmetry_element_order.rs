use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::aux::misc::HashableFloat;


/// An enum to handle symmetry element orders which can be integers, floats, or infinity.
#[derive(Clone, Debug)]
pub enum ElementOrder {
    /// Positive integer order.
    Int(u32),

    /// Positive floating point order and a threshold for comparisons.
    Float(f64, f64),

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
        Self::Float(rounded_order, thresh)
    }

    pub fn to_float(&self) -> f64 {
        match self {
            Self::Int(s_i) => *s_i as f64,
            Self::Float(s_f, _) => *s_f,
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
                Self::Float(o_f, o_thresh) => {
                    return approx::relative_eq!(
                        *s_i as f64,
                        *o_f,
                        epsilon = *o_thresh,
                        max_relative = *o_thresh
                    );
                }
                Self::Inf => return false,
            },
            Self::Float(s_f, s_thresh) => match &other {
                Self::Int(o_i) => {
                    return approx::relative_eq!(
                        *s_f,
                        *o_i as f64,
                        epsilon = *s_thresh,
                        max_relative = *s_thresh
                    );
                }
                Self::Float(o_f, o_thresh) => {
                    return approx::relative_eq!(
                        *s_f,
                        *o_f,
                        epsilon = (*s_thresh * *o_thresh).sqrt(),
                        max_relative = (*s_thresh * *o_thresh).sqrt(),
                    );
                }
                Self::Inf => {
                    return s_f.is_infinite() && s_f.is_sign_positive();
                }
            },
            Self::Inf => match &other {
                Self::Int(_) => return false,
                Self::Float(o_f, _) => {
                    return o_f.is_infinite() && o_f.is_sign_positive();
                }
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
            Self::Float(s_f, s_thresh) => {
                let factor = 1.0 / s_thresh;
                s_f.round_factor(factor).integer_decode().hash(state);
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
            Self::Float(s_f, s_thresh) => {
                match approx::relative_eq!(
                    *s_f,
                    s_f.round(),
                    epsilon = *s_thresh,
                    max_relative = *s_thresh
                ) {
                    true => write!(f, "{:.0}", s_f),
                    false => write!(f, "{:.3}", s_f),
                }
            }
            Self::Inf => write!(f, "{}", "∞".to_owned()),
        }
    }
}

pub const ORDER_1: ElementOrder = ElementOrder::Int(1);
pub const ORDER_2: ElementOrder = ElementOrder::Int(2);
pub const ORDER_I: ElementOrder = ElementOrder::Inf;

