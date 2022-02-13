use std::fmt;
use nalgebra as na;
use std::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug)]
pub struct Point3D<T> {
    pub x: T,
    pub y: T,
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

impl<T: Sub<Output = T> + Copy> Sub<&Point3D<T>> for Point3D<T> {
    type Output = Self;
    fn sub(self, other: &Point3D<T>) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: Mul<Output = T> + Copy> Mul<T> for &Point3D<T> {
    type Output = Point3D<T>;
    fn mul(self, other: T) -> Point3D<T> {
        Point3D {
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
    pub fn sq_norm(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    pub fn norm(&self) -> T {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}
