use byteorder::LittleEndian;
use ndarray::{array, Array2};

use crate::io::numeric::NumericReader;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

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
