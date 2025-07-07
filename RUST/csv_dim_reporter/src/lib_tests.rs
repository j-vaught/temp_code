use radar_to_netcdf::lib::*;
use chrono::{FixedOffset, TimeZone};

#[test]
fn test_parse_info() {
    let md = parse_info("tests/data/info.txt".as_ref()).unwrap();
    assert_eq!(&md["Metadata"]["latitude"], "34.06");
    assert_eq!(&md["Metadata"]["heading"], "123.4");
}

#[test]
fn test_list_sweeps() {
    let sweeps = list_sweeps("tests/data".as_ref()).unwrap();
    assert_eq!(sweeps.len(), 2);
    assert!(sweeps[0].ends_with("20250101_000000_000.csv"));
}

#[test]
fn test_parse_filename_to_secs() {
    let tz = FixedOffset::east_opt(0).unwrap();
    let secs1 = parse_filename_to_secs("20250101_000000_000.csv", tz).unwrap();
    let secs2 = parse_filename_to_secs("20250101_000100_500.csv", tz).unwrap();
    assert!(secs2 > secs1);
}

#[test]
fn test_parse_csv_record() {
    // Make a dummy StringRecord: "0,10,1,50,4096,<873 zeros>"
    let mut vals = vec!["0".to_string(), "10","1","50","4096"]
        .into_iter().map(String::from).collect::<Vec<_>>();
    vals.extend(std::iter::repeat("100.0".to_string()).take(GATE));
    let rec = csv::StringRecord::from(vals);
    let (sc, rc, gn, azi, e) = parse_csv_record(&rec).unwrap();
    assert_eq!(sc, 10);
    assert_eq!(rc, 1);
    assert_eq!(gn, 50);
    assert!((azi - 180.0).abs() < 1e-6);
    assert_eq!(e.len(), GATE);
}
