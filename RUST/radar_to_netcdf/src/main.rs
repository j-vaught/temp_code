
// USAGE cargo run --release /path/to/data /path/to/output.nc

use anyhow::{anyhow, bail, Result};
use chrono::{Duration, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime, TimeZone};
use csv::StringRecord;
use glob::glob;
use netcdf;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::Instant,
};

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────
pub const MAX_PULSE: usize = 8192;
pub const GATE:      usize = 868;
pub const FILL_F32:  f32   = -9_999.0;

// ─────────────────────────────────────────────────────────────────────
// Simple timing helper
// ─────────────────────────────────────────────────────────────────────
fn timeit<T, F: FnOnce() -> T>(label: &str, f: F) -> T {
    let t0 = Instant::now();
    let out = f();
    eprintln!("{label:<20}{:?}", t0.elapsed());
    out
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────
pub fn parse_info(path: &Path) -> Result<HashMap<String, HashMap<String, String>>> {
    let rdr = BufReader::new(File::open(path)?);
    let mut out = HashMap::<String, HashMap<String, String>>::new();
    for line in rdr.lines() {
        let t = line?.trim().to_string();
        if t.is_empty() || t.starts_with('#') { continue }
        let parts: Vec<_> = t.split(',').map(str::to_string).collect();
        if parts.len() != 3 { return Err(anyhow!("Malformed: {}", t)); }
        out.entry(parts[0].clone()).or_default().insert(parts[1].clone(), parts[2].clone());
    }
    Ok(out)
}

pub fn list_sweeps(base: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<_> = glob(&format!("{}/good/*.csv", base.display()))?
        .filter_map(Result::ok)
        .collect();
    v.sort();
    Ok(v)
}

// ─────────────────────────────────────────────────────────────────────
// CSV → buffers
// ─────────────────────────────────────────────────────────────────────
fn fill_buffers_from_csv(
    csv_path:  &Path,
    echo_buf:  &mut [f32],
    angle_buf: &mut [f32],
) -> Result<(i32, i32, i32)> {
    if echo_buf.len() != MAX_PULSE * GATE {
        bail!("echo buffer size mismatch, expected {}×{}", MAX_PULSE, GATE);
    }
    if angle_buf.len() != MAX_PULSE {
        bail!("angle buffer size mismatch, expected {}", MAX_PULSE);
    }

    echo_buf.fill(FILL_F32);
    angle_buf.fill(FILL_F32);

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(csv_path)?;

    let records = rdr.records().skip(1);
    let mut scale: Option<i32> = None;
    let mut range: Option<i32> = None;
    let mut gain:  Option<i32> = None;

    for (p, result) in records.enumerate() {
        if p >= MAX_PULSE { break; }
        let rec = result?;
        let (sc, rc, gn, azi, echos) = parse_csv_record(&rec)?;

        scale.get_or_insert(sc);
        range.get_or_insert(rc);
        gain .get_or_insert(gn);

        angle_buf[p] = azi;
        let base = p * GATE;
        echo_buf[base..base + GATE].copy_from_slice(&echos);
    }

    Ok((scale.unwrap_or_default(), range.unwrap_or_default(), gain.unwrap_or_default()))
}

fn parse_csv_record(rec: &StringRecord) -> Result<(i32, i32, i32, f32, Vec<f32>)> {
    if rec.len() < 5 + GATE {
        bail!("row too short: expected at least {} fields, got {}", 5 + GATE, rec.len());
    }

    let sc:  i32 = rec[1].parse()?;
    let rc:  i32 = rec[2].parse()?;
    let gn:  i32 = rec[3].parse()?;
    let ang: i32 = rec[4].parse()?;
    let azi       = ang as f32 / 8192.0 * 360.0; // 13-bit encoder → degrees

    let mut echos = Vec::with_capacity(GATE);
    for field in rec.iter().skip(5).take(GATE) { echos.push(field.parse()?); }

    Ok((sc, rc, gn, azi, echos))
}

// ─────────────────────────────────────────────────────────────────────
// File-name → epoch seconds  (manual error wrap)
// ─────────────────────────────────────────────────────────────────────
fn parse_filename_to_secs(fname: &str, tz: FixedOffset) -> Result<f64> {
    let stem = Path::new(fname)
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("bad fname"))?;
    let parts: Vec<_> = stem.split('_').collect();
    if parts.len() != 3 { return Err(anyhow!("unexpected: {stem}")); }

    let date = NaiveDate::parse_from_str(parts[0], "%Y%m%d")
        .map_err(|e| anyhow!("date parse error: {e}"))?;
    let time = NaiveTime::parse_from_str(parts[1], "%H%M%S")
        .map_err(|e| anyhow!("time parse error: {e}"))?;

    let msec: u32 = parts[2].parse()?;
    let naive = NaiveDateTime::new(date, time) + Duration::milliseconds(msec as i64);
    let dt = tz.from_local_datetime(&naive).single().ok_or_else(|| anyhow!("ambiguous"))?;
    Ok(dt.timestamp() as f64 + f64::from(dt.timestamp_subsec_micros()) / 1e6)
}

// ─────────────────────────────────────────────────────────────────────
// In-memory representation of one sweep
// ─────────────────────────────────────────────────────────────────────
struct SweepData {
    idx:   usize,
    time:  f64,
    echo:  Vec<f32>,
    angle: Vec<f32>,
    scale: i32,
    range: i32,
    gain:  i32,
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────
fn main() -> Result<()> {
    let base_dir    = Path::new("/Users/jacobvaught/Downloads/Research_radar_DATA/data/data_9");
    let output_path = "/Users/jacobvaught/Downloads/Research_radar_DATA/data/data_9/output.nc";
    let _ = std::fs::remove_file(output_path);

    let sweeps = timeit("list_sweeps", || list_sweeps(base_dir))?;
    let tz = FixedOffset::east_opt(0).unwrap();

    // ─────────────────────────────────────────────────────────────────────
    // 1.  **Create the NetCDF container (unchanged)**
    // ─────────────────────────────────────────────────────────────────────
    let mut nc = timeit("create_netcdf", || -> Result<_> {
        let mut nc = netcdf::create(output_path)?;
        nc.add_unlimited_dimension("time")?;
        nc.add_dimension("pulse", MAX_PULSE)?;
        nc.add_dimension("gate",  GATE)?;

        nc.add_variable::<f64>("time",  &["time"])?;

        {
            let mut v = nc.add_variable::<f32>("echo",  &["time", "pulse", "gate"])?;
            v.set_chunking(&[1, MAX_PULSE, GATE])?;
            v.set_compression(9, true)?;
            v.set_fill_value(FILL_F32)?;
        }
        {
            let mut v = nc.add_variable::<f32>("angle", &["time", "pulse"])?;
            v.set_fill_value(FILL_F32)?;
        }

        nc.add_variable::<i32>("scale", &["time"])?;
        nc.add_variable::<i32>("range", &["time"])?;
        nc.add_variable::<i32>("gain",  &["time"])?;

        Ok(nc)
    })?;

    // ─────────────────────────────────────────────────────────────────────
    // 2.  **Parse each sweep in small parallel batches and write them out**
    //     — keeps memory bounded while preserving rayon parallelism
    // ─────────────────────────────────────────────────────────────────────
    const BATCH: usize = 32;               // ~32 × 28 MiB ≈ 0.9 GiB resident

    use chrono::{Local, Timelike};
    timeit("parse+write_batches", || -> Result<()> {
    // ---------------- progress bookkeeping ----------------
    let total_batches = (sweeps.len() + BATCH - 1) / BATCH;

    for (batch_no, chunk) in sweeps.chunks(BATCH).enumerate() {
        // ----------- live heartbeat every 10th batch (and the last) ------
        if batch_no % 10 == 0 || batch_no + 1 == total_batches {
            println!(
                "[{:02}:{:02}:{:02}]  batch {}/{}  ({} sweeps parsed so far)",
                chrono::Local::now().hour(),
                chrono::Local::now().minute(),
                chrono::Local::now().second(),
                batch_no + 1,
                total_batches,
                (batch_no + 1).saturating_mul(BATCH).min(sweeps.len())
            );
        }

        // ----------- PARSE this chunk in parallel ------------------------
        let sweep_data: Vec<SweepData> = chunk
            .par_iter()
            .enumerate()                                  // offset inside chunk
            .map(|(off, path)| -> Result<SweepData> {
                // tiny breadcrumb whenever a new file kicks off
                if off == 0 {
                    eprintln!("  └─ parsing {:?}", path.display());
                }

                let idx  = batch_no * BATCH + off;        // global index
                let time = parse_filename_to_secs(
                    path.file_name().unwrap().to_str().unwrap(), tz)?;

                let mut echo  = vec![FILL_F32; MAX_PULSE * GATE];
                let mut angle = vec![FILL_F32; MAX_PULSE];
                let (scale, range, gain) =
                    fill_buffers_from_csv(path, &mut echo, &mut angle)?;

                Ok(SweepData { idx, time, echo, angle, scale, range, gain })
            })
            .collect::<Result<_>>()?;                     // ≤ BATCH items live here

        // ----------- WRITE this chunk sequentially -----------------------
        for s in &sweep_data {
            nc.variable_mut("time" ).unwrap().put_values(&[s.time ], (s.idx,))?;
            nc.variable_mut("echo" ).unwrap()
                .put_values(&s.echo , (&[s.idx, 0, 0], &[1, MAX_PULSE, GATE]))?;
            nc.variable_mut("angle").unwrap()
                .put_values(&s.angle, (&[s.idx, 0], &[1, MAX_PULSE]))?;
            nc.variable_mut("scale").unwrap().put_values(&[s.scale], (s.idx,))?;
            nc.variable_mut("range").unwrap().put_values(&[s.range], (s.idx,))?;
            nc.variable_mut("gain" ).unwrap().put_values(&[s.gain ], (s.idx,))?;
        }                              // sweep_data drops here → memory freed
    }
    Ok(())
})?;



    // ------------ optional global attributes --------------------------------
    if let Ok(info_map) = parse_info(&base_dir.join("info.txt")) {
        for (section, kv) in info_map {
            for (key, value) in kv {
                nc.add_attribute(&format!("{}_{}", section, key), value.as_str())?;
            }
        }
    }

    drop(nc);
    eprintln!("Finished OK, wrote {} sweeps → {output_path}", sweeps.len());

    Ok(())
}