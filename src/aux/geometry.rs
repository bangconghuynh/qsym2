use nalgebra as na;
use std::fmt;
use std::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};

/// A struct representing a point in three dimensions.
#[derive(Debug)]
pub struct Point3D<T> {
    /// The $x$-component of the coordinates of the point.
    pub x: T,
    /// The $y$-component of the coordinates of the point.
    pub y: T,
    /// The $z$-component of the coordinates of the point.
    pub z: T,
}

impl<T: fmt::Display> fmt::Display for Point3D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.6}, {:.6}, {:.6})", self.x, self.y, self.z)
    }
}

impl<T: Add<Output = T>> Add for Point3D<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: Sub<Output = T>> Sub for Point3D<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: Add<Output = T> + Copy> Add<&Point3D<T>> for &Point3D<T> {
    type Output = Point3D<T>;
    fn add(self, other: &Point3D<T>) -> Self::Output {
        Self::Output {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: Sub<Output = T> + Copy> Sub<&Point3D<T>> for &Point3D<T> {
    type Output = Point3D<T>;
    fn sub(self, other: &Point3D<T>) -> Self::Output {
        Self::Output {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: Mul<Output = T> + Copy> Mul<T> for &Point3D<T> {
    type Output = Point3D<T>;
    fn mul(self, other: T) -> Self::Output {
        Self::Output {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl<T: Neg<Output = T>> Neg for Point3D<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<T: AddAssign> AddAssign for Point3D<T> {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<T: SubAssign + Copy> SubAssign<&Point3D<T>> for Point3D<T> {
    fn sub_assign(&mut self, other: &Point3D<T>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<T: MulAssign + Copy> MulAssign<T> for Point3D<T> {
    fn mul_assign(&mut self, other: T) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
    }
}

impl<T: Copy> Index<usize> for Point3D<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Point3D index out of bound."),
        }
    }
}

impl<T: na::ComplexField + Copy> Point3D<T> {
    /// Calculates the squared norm of the point taken as a position vector.
    pub fn sq_norm(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Calculates the norm of the point taken as a position vector.
    pub fn norm(&self) -> T {
        (self.sq_norm()).sqrt()
    }
}
