use byteorder::LittleEndian;
use ndarray::{array, Array2};
use num_complex::Complex;

use crate::io::numeric::NumericReader;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

type C128 = Complex<f64>;

#[test]
fn test_io_numeric_reader_f64() {
    let path = format!("{ROOT}/tests/binaries/nine_f64");
    let v = NumericReader::<_, LittleEndian, f64>::from_file(path)
        .unwrap()
        .collect::<Vec<_>>();
    let a = Array2::from_shape_vec((3, 3), v).unwrap();
    let a_ref = array![
        [0.0f64, 1.0f64, 2.0f64],
        [3.0f64, 4.0f64, 5.0f64],
        [6.0f64, 7.0f64, 8.0f64],
    ];
    assert_eq!(a, a_ref);
}

#[test]
fn test_io_numeric_reader_i32() {
    let path = format!("{ROOT}/tests/binaries/fifteen_i32");
    let v = NumericReader::<_, LittleEndian, i32>::from_file(path)
        .unwrap()
        .collect::<Vec<_>>();
    let a = Array2::from_shape_vec((3, 5), v).unwrap();
    let a_ref = array![[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14],];
    assert_eq!(a, a_ref);
}

#[test]
fn test_io_numeric_reader_i64() {
    let path = format!("{ROOT}/tests/binaries/fifteen_i64");
    let v = NumericReader::<_, LittleEndian, i64>::from_file(path)
        .unwrap()
        .collect::<Vec<_>>();
    let a = Array2::from_shape_vec((3, 5), v).unwrap();
    let a_ref = array![[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14],];
    assert_eq!(a, a_ref);
}

#[test]
fn test_io_numeric_reader_c128() {
    let path = format!("{ROOT}/tests/binaries/sixteen_c128");
    let v = NumericReader::<_, LittleEndian, Complex<f64>>::from_file(path)
        .unwrap()
        .collect::<Vec<_>>();
    let a = Array2::from_shape_vec((4, 4), v).unwrap();
    let a_ref = array![
        [C128::new( 0.0, 16.0), C128::new( 1.0, 15.0), C128::new(2.0, 14.0), C128::new( 3.0, 13.0)],
        [C128::new( 4.0, 12.0), C128::new( 5.0, 11.0), C128::new(6.0, 10.0), C128::new( 7.0,  9.0)],
        [C128::new( 8.0,  8.0), C128::new( 9.0,  7.0), C128::new(10.0, 6.0), C128::new(11.0,  5.0)],
        [C128::new(12.0,  4.0), C128::new(13.0,  3.0), C128::new(14.0, 2.0), C128::new(15.0,  1.0)],
    ];
    assert_eq!(a, a_ref);
}
