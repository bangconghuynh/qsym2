use std::fs::File;
use std::io::{BufRead, BufReader};
use std::marker::PhantomData;
use std::path::Path;

use anyhow;
use byteorder::{BigEndian, ByteOrder, LittleEndian};

#[cfg(test)]
#[path = "numeric_tests.rs"]
mod numeric_tests;

/// An iterable structure for reading numeric binary files.
struct NumericReader<R: BufRead, B: ByteOrder, T> {
    /// The inner file reader.
    inner: R,

    /// The byte order of the numeric values to be read.
    byte_order: PhantomData<B>,

    /// The type of the numeric values to be read.
    numeric_type: PhantomData<T>,
}

impl<R: BufRead, B: ByteOrder, T> NumericReader<R, B, T> {
    /// Constructs a new numeric binary reader wrapping around a standard file reader.
    ///
    /// # Arguments
    ///
    /// * `inner` - The underlying file reader.
    fn new(inner: R) -> Self {
        Self {
            inner,
            byte_order: PhantomData,
            numeric_type: PhantomData,
        }
    }
}

impl<B: ByteOrder, T> NumericReader<BufReader<File>, B, T> {
    /// Constructs a new numeric binary reader wrapping around a buffered file reader from a
    /// filename.
    ///
    /// # Arguments
    ///
    /// * `filename` - The path to the file to be read.
    fn from_file<P: AsRef<Path>>(filename: P) -> Result<Self, anyhow::Error> {
        let f = File::open(&filename)?;
        Ok(Self::new(BufReader::new(f)))
    }
}

macro_rules! impl_iterator_numeric_reader {
    ($($t:ty),+) => {$(
        impl<R: BufRead> Iterator for NumericReader<R, LittleEndian, $t> {
            type Item = $t;

            fn next(&mut self) -> Option<Self::Item> {
                let mut buff: [u8; std::mem::size_of::<$t>()] = [0_u8; std::mem::size_of::<$t>()];
                self.inner.read_exact(&mut buff).ok()?;
                Some(<$t>::from_le_bytes(buff))
            }
        }

        impl<R: BufRead> Iterator for NumericReader<R, BigEndian, $t> {
            type Item = $t;

            fn next(&mut self) -> Option<Self::Item> {
                let mut buff: [u8; std::mem::size_of::<$t>()] = [0_u8; std::mem::size_of::<$t>()];
                self.inner.read_exact(&mut buff).ok()?;
                Some(<$t>::from_be_bytes(buff))
            }
        }
    )+}
}

impl_iterator_numeric_reader!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64
);
