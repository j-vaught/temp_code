// usage is to do cargo run --release path/to/directory
// where directory contains csv files with duplicates(aka interpolation and Rezboost(tm) turned off).


use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
use csv::{ReaderBuilder, WriterBuilder};

// Entry point, same as before.
fn main() -> Result<(), Box<dyn Error>> {
    let dir = env::args()
        .nth(1)
        .expect("Usage: csv_duplicate_fixer <directory_path>");
    let base_dir = Path::new(&dir);

    let csv_files = collect_csvs(base_dir)?;

    for path in csv_files {
        match process_csv_both_versions(&path) {
            Ok((info_first, info_second)) => {
                println!(
                    "{}: {} lines (keep first), {} records, {} duplicates removed",
                    path.display(), info_first.total_lines, info_first.data_rows, info_first.removed
                );
                println!(
                    "{}: {} lines (keep second), {} records, {} duplicates removed (saved as '{}')",
                    path.display(), info_second.total_lines, info_second.data_rows, info_second.removed,
                    make_a_path(&path).display()
                );
            }
            Err(e) => eprintln!("Error in {}: {}", path.display(), e),
        }
    }

    Ok(())
}

struct ProcessInfo {
    total_lines: usize,
    data_rows: usize,
    removed: usize,
}

fn collect_csvs(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut list = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension()
               .and_then(|e| e.to_str())
               .map_or(false, |e| e.eq_ignore_ascii_case("csv")) {
            list.push(path);
        }
    }
    Ok(list)
}

// Utility to generate the next filename, e.g. myfile_079.csv -> myfile_080.csv
fn make_a_path(path: &Path) -> PathBuf {
    let file_stem = path.file_stem().unwrap().to_string_lossy();
    let ext = path.extension().unwrap_or_default();

    let parts: Vec<&str> = file_stem.split('_').collect();
    if let Some(last_part) = parts.last() {
        if let Ok(num) = last_part.parse::<u32>() {
            let next_num = num + 1;
            let new_last_part = format!("{:03}", next_num); // Ensure 3 digits, e.g., 080

            let new_file_stem = if parts.len() > 1 {
                format!("{}_{}", parts[0..parts.len() - 1].join("_"), new_last_part)
            } else {
                new_last_part
            };

            let mut new_path = path.with_file_name(new_file_stem);
            new_path.set_extension(ext);
            return new_path;
        }
    }

    // Fallback if the naming convention doesn't match, append 'a' as before
    let mut filename = path.file_stem().unwrap().to_os_string();
    filename.push("a");
    let mut new_path = path.with_file_name(filename);
    new_path.set_extension(ext);
    new_path
}


// Handles both versions (keep first and keep second duplicates)
fn process_csv_both_versions(path: &Path) -> Result<(ProcessInfo, ProcessInfo), Box<dyn Error>> {
    // Read lines
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut lines = Vec::new();
    for line in reader.lines() {
        lines.push(line?);
    }
    let total_lines = lines.len();

    // Short files, do nothing
    if total_lines < 2 {
        return Ok((
            ProcessInfo { total_lines, data_rows: 0, removed: 0 },
            ProcessInfo { total_lines, data_rows: 0, removed: 0 }
        ));
    }

    // Remove and store the first line (ID/identification)
    let id_line = lines.remove(0);

    // Parse the CSV rows into a Vec<Vec<String>>
    let blob = lines.join("\n");
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(blob.as_bytes());
    let data: Vec<Vec<String>> = rdr
        .records()
        .map(|r| r.map(|rec| rec.iter().map(|s| s.to_string()).collect::<Vec<_>>()).map_err(Into::into))
        .collect::<Result<_, Box<dyn Error>>>()?;

    // Get filtered versions
    let (keep_first, removed_first) = filter_duplicates(&data, true);
    let (keep_second, removed_second) = filter_duplicates(&data, false);

    // Write back "keep first" to the original file
    {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(&mut writer, "{}", id_line)?;
        let mut wtr = WriterBuilder::new().has_headers(false).from_writer(writer);
        for row in &keep_first {
            wtr.write_record(row)?;
        }
        wtr.flush()?;
    }

    // Write "keep second" to 'a' file
    {
        let a_path = make_a_path(path);
        let file = File::create(a_path)?;
        let mut writer = BufWriter::new(file);
        writeln!(&mut writer, "{}", id_line)?;
        let mut wtr = WriterBuilder::new().has_headers(false).from_writer(writer);
        for row in &keep_second {
            wtr.write_record(row)?;
        }
        wtr.flush()?;
    }

    Ok((
        ProcessInfo { total_lines: total_lines, data_rows: keep_first.len(), removed: removed_first },
        ProcessInfo { total_lines: total_lines, data_rows: keep_second.len(), removed: removed_second }
    ))
}

// Filters out duplicates in the 5th column (index 4).
// keep_first: if true, keeps the first duplicate and removes the second; else, keeps the second.
fn filter_duplicates(data: &[Vec<String>], keep_first: bool) -> (Vec<Vec<String>>, usize) {
    let mut out = Vec::new();
    let mut prev_val: Option<&str> = None;
    let mut skip_next = false;
    let mut removed = 0;

    let mut i = 0;
    while i < data.len() {
        let current_val = data[i][4].as_str();
        if prev_val == Some(current_val) {
            // This is a duplicate
            if keep_first {
                // skip this duplicate (keep the first occurrence)
                removed += 1;
                i += 1;
                continue;
            } else {
                // keep this duplicate (remove the first one we already pushed)
                if !out.is_empty() {
                    out.pop(); // remove previous
                    removed += 1;
                }
                out.push(data[i].clone());
                prev_val = Some(current_val);
                i += 1;
                continue;
            }
        } else {
            out.push(data[i].clone());
            prev_val = Some(current_val);
            i += 1;
        }
    }

    (out, removed)
}
