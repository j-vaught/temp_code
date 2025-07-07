// src/main.rs

// Classifies radar CSV files in a given directory into three categories: Bad, Gain, and Good, based on two checks:
// 1) Whether the values in column 5 are strictly increasing (excluding the first and last lines).
// 2) Whether columns 1, 2, and 3 are consistent across all rows (excluding the first and last lines).
// It summarizes the number and average size of files in each category, then optionally moves them into subfolders (bad/, gain_change/, good/) based on user confirmation.
// Supports an optional `--debug` flag (only in debug builds) to log per-row checks.

// USAGE #### cargo run --release -- <directory_path>

use chrono::{Local, DateTime};
use csv::StringRecord;
use rayon::prelude::*;

use std::{
    env,           // CLI args
    error::Error,  // boxed error trait
    fs,            // filesystem operations
    io::{stdin, stdout, Write}, // interactive prompt
    path::{Path, PathBuf},      // Path manipulation
};

/// Category assigned to each file after classification
#[derive(Debug, Clone, Copy)]
enum Category {
    Bad,
    Gain,
    Good,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Log program start
    let start: DateTime<Local> = Local::now();
    println!("[{}] Starting CSV classification…", start.format("%Y-%m-%d %H:%M:%S"));

    // Read target directory + optional --debug flag (only active in debug builds)
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <directory_path> [--debug]", args[0]);
        std::process::exit(1);
    }
    let dir_arg = &args[1];
    // This is true only in a debug build AND if you pass --debug or -d
    let debug = cfg!(debug_assertions) && args.iter().any(|a| a == "--debug" || a == "-d");

    let base_dir = Path::new(&dir_arg);

    // Collect all CSV files
    let all_files = collect_csvs(base_dir)?;

    // Parallel classification: (PathBuf, Category)
    let classified: Vec<(PathBuf, Category)> = all_files
        .par_iter()
        .filter_map(|path| {
            if debug {
                println!("[DEBUG] Processing ▶ {}", path.display());
            }
            // Phase 1: strict increase on column 5
            match check_strict_increase(path, debug) {
                Ok(true) => {
                    // Phase 2: gain consistency on cols 1–3
                    match scan_gain_consistency(path, debug) {
                        Ok(true) => Some((path.clone(), Category::Good)),
                        Ok(false) => Some((path.clone(), Category::Gain)),
                        Err(e) => { log_error(path, &e, "gain scan error"); None }
                    }
                }
                Ok(false) => Some((path.clone(), Category::Bad)),
                Err(e) => { log_error(path, &e, "strict check error"); None }
            }
        })
        .collect();

    // Group by category
    let mut bad_files: Vec<PathBuf> = Vec::new();
    let mut gain_files: Vec<PathBuf> = Vec::new();
    let mut good_files: Vec<PathBuf> = Vec::new();
    for (path, cat) in &classified {
        match cat {
            Category::Bad  => bad_files.push(path.clone()),
            Category::Gain => gain_files.push(path.clone()),
            Category::Good => good_files.push(path.clone()),
        }
    }

    // Summarize
    let (bad_count, bad_avg)   = summarize(&bad_files)?;
    let (gain_count, gain_avg) = summarize(&gain_files)?;
    let (good_count, good_avg) = summarize(&good_files)?;

    println!("\nSummary:");
    println!("  Bad:  {} files, avg size {} bytes", bad_count, bad_avg);
    println!("  Gain: {} files, avg size {} bytes", gain_count, gain_avg);
    println!("  Good: {} files, avg size {} bytes", good_count, good_avg);

    // Prompt for moving
    print!("Move files into bad/, gain_change/, good/? (y/N): ");
    stdout().flush()?;
    let mut input = String::new();
    stdin().read_line(&mut input)?;
    let resp = input.trim().to_lowercase();
    if resp == "y" || resp == "yes" {
        // Create target dirs
        let bad_dir  = base_dir.join("bad");
        let gain_dir = base_dir.join("gain_change");
        let good_dir = base_dir.join("good");
        fs::create_dir_all(&bad_dir)?;
        fs::create_dir_all(&gain_dir)?;
        fs::create_dir_all(&good_dir)?;

        // Move files by category
        bad_files.iter().for_each(|p| move_file(p, &bad_dir, "bad"));
        gain_files.iter().for_each(|p| move_file(p, &gain_dir, "gain_change"));
        good_files.iter().for_each(|p| move_file(p, &good_dir, "good"));
    } else {
        println!("No files were moved.");
    }

    // Log program end
    let end     : DateTime<Local> = Local::now();
    let elapsed = end.signed_duration_since(start).num_seconds();
    println!("[{}] Finished. Total time: {}s", end.format("%Y-%m-%d %H:%M:%S"), elapsed);

    Ok(())
}

/// Collects all `.csv` files directly in the given directory
fn collect_csvs(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let list = fs::read_dir(dir)?
        .filter_map(|e| e.ok()
            .and_then(|ent| ent.path().extension()
                .and_then(|ext| ext.to_str())
                .filter(|&ext| ext.eq_ignore_ascii_case("csv"))
                .map(|_| ent.path())
            )
        )
        .collect();
    Ok(list)
}

/// Summarizes file count and average size (in bytes)
fn summarize(files: &[PathBuf]) -> Result<(usize, u64), Box<dyn Error>> {
    let count = files.len();
    let total: u64 = files.iter()
        .filter_map(|p| fs::metadata(p).ok().map(|m| m.len()))
        .sum();
    let avg = if count > 0 { total / count as u64 } else { 0 };
    Ok((count, avg))
}

/// Moves `path` into `target_dir` with logging
fn move_file(path: &Path, target_dir: &Path, category: &str) {
    if let Err(e) = fs::rename(path, &target_dir.join(path.file_name().unwrap())) {
        eprintln!("Failed to move {} to {}: {}", path.display(), category, e);
    }
}

/// Logs a parse or I/O error with timestamp
fn log_error(path: &Path, err: &Box<dyn Error>, phase: &str) {
    let now: DateTime<Local> = Local::now();
    eprintln!("[{}] {}: {}: {}", now.format("%Y-%m-%d %H:%M:%S"), path.display(), phase, err);
}

/// Checks if column 5 (index 4) is strictly increasing (skip first & last lines)
fn check_strict_increase(path: &Path, debug: bool) -> Result<bool, Box<dyn Error>> {
    if debug {
        println!("[DEBUG]   → check_strict_increase for {}", path.display());
    }
    let content = fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < 3 {
        return Ok(true);
    }

    let csv_text = lines[1..lines.len()-1].join("\n");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(csv_text.as_bytes());
    let records = rdr.records()
        .map(|r| r.map_err(|e| Box::<dyn Error>::from(e)))
        .collect::<Result<Vec<StringRecord>, _>>()?;
    if records.len() < 2 {
        return Ok(true);
    }

    for i in 1..records.len() {
        let prev: f64 = records[i-1].get(4).unwrap_or("nan").parse()?;
        let curr: f64 = records[i].get(4).unwrap_or("nan").parse()?;
        if debug {
            println!("      [DEBUG] row {:>4}: prev = {:>10}, curr = {:>10}", i, prev, curr);
        }
        if curr <= prev {
            if debug {
                println!("      [DEBUG]   ✗ violation at row {}: {} ≤ {}", i, curr, prev);
            }
            return Ok(false);
        }
    }

    Ok(true)
}

/// Scans columns 1, 2, and 3 for consistency (skip first & last lines)
fn scan_gain_consistency(path: &Path, debug: bool) -> Result<bool, Box<dyn Error>> {
    if debug {
        println!("[DEBUG]   → scan_gain_consistency for {}", path.display());
    }
    let content = fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < 3 {
        return Ok(true);
    }

    let csv_blob = lines[1..lines.len()-1].join("\n");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(csv_blob.as_bytes());
    let records = rdr.records()
        .map(|r| r.map_err(|e| Box::<dyn Error>::from(e)))
        .collect::<Result<Vec<StringRecord>, _>>()?;
    if records.is_empty() {
        return Ok(true);
    }

    let base1 = records[0].get(1).unwrap_or("").to_string();
    let base2 = records[0].get(2).unwrap_or("").to_string();
    let base3 = records[0].get(3).unwrap_or("").to_string();

    for (idx, rec) in records.iter().skip(1).enumerate() {
        let v1 = rec.get(1).unwrap_or("");
        let v2 = rec.get(2).unwrap_or("");
        let v3 = rec.get(3).unwrap_or("");
        if debug {
            println!(
                "      [DEBUG] compare row {} → ({},{},{}) vs base ({},{},{})",
                idx + 2,  // +2 because we skipped the first record and header line
                v1, v2, v3,
                base1, base2, base3
            );
        }
        if v1 != base1 || v2 != base2 || v3 != base3 {
            if debug {
                println!("      [DEBUG]   ✗ mismatch at row {}!", idx + 2);
            }
            return Ok(false);
        }
    }

    Ok(true)
}
