use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime};
use csv::StringRecord;
use glob::glob;
use std::{
    collections::HashMap,
    fs::File,
    io::BufRead,
    io::BufReader,
    path::{Path, PathBuf},
};

pub const MAX_PULSE: usize = 8192;
pub const GATE: usize      = 873;

/// 1. Parse info.txt into group→(key→value) map
pub fn parse_info(path: &Path) -> Result<HashMap<String, HashMap<String, String>>> {
    let rdr = BufReader::new(File::open(path)?);
    let mut out = HashMap::new();
    for line in rdr.lines() {
        let t = line?.trim().to_string();
        if t.is_empty() || t.starts_with('#') { continue }
        let parts: Vec<_> = t.split(',').map(str::to_string).collect();
        if parts.len() != 3 {
            return Err(anyhow!("Malformed: {}", t));
        }
        out.entry(parts[0].clone())
           .or_default()
           .insert(parts[1].clone(), parts[2].clone());
    }
    Ok(out)
}

/// 2. Find all good/*.csv under `base`, sorted lexically
pub fn list_sweeps(base: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<_> = glob(&format!("{}/good/*.csv", base.display()))?
        .filter_map(Result::ok)
        .collect();
    v.sort();
    Ok(v)
}

/// 3. Turn `YYYYMMDD_HHMMSS_mmm.csv` → seconds since epoch
pub fn parse_filename_to_secs(fname: &str, tz: FixedOffset) -> Result<f64> {
    let stem = Path::new(fname)
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("bad fname"))?;
    let parts: Vec<_> = stem.split('_').collect();
    if parts.len() != 3 {
        return Err(anyhow!("unexpected: {}", stem));
    }
    let date = NaiveDate::parse_from_str(parts[0], "%Y%m%d")?;
    let time = NaiveTime::parse_from_str(parts[1], "%H%M%S")?;
    let msec: u32 = parts[2].parse()?;
    let naive = NaiveDateTime::new(date, time) + Duration::milliseconds(msec as i64);
    let dt = tz.from_local_datetime(&naive)
               .single()
               .ok_or_else(|| anyhow!("ambiguous"))?;
    Ok(dt.timestamp() as f64 + f64::from(dt.timestamp_subsec_micros())/1e6)
}

/// 4. Parse one CSV record (a `StringRecord`) into your five buffers
pub fn parse_csv_record(rec: &StringRecord) -> Result<(i32,i32,i32,f32, Vec<f32>)> {
    let sc: i32 = rec[1].parse()?;
    let rc: i32 = rec[2].parse()?;
    let gn: i32 = rec[3].parse()?;
    let ang: i32= rec[4].parse()?;
    let azi = ang as f32 / 8192.0 * 360.0;
    let mut echos = Vec::with_capacity(GATE);
    for i in 5..5+GATE {
        echos.push(rec[i].parse()?);
    }
    Ok((sc, rc, gn, azi, echos))
}
