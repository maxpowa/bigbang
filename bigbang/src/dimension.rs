use std::fmt::Display;
use serde::{Deserialize, Serialize};
#[derive(Clone, PartialEq, Serialize, Deserialize)]
/// Used to represent which dimension the GravTree node has split on.
pub enum Dimension {
    X,
    Y,
    Z,
}

/// Convenience function that returns the Dimension as a &str.
impl Display for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match *self {
            Dimension::X => String::from("X"),
            Dimension::Y => String::from("Y"),
            Dimension::Z => String::from("Z"),
        };
        write!(f, "{}", str)
    }
}
