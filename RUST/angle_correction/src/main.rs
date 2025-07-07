//fixes errors with angles in CSV files when interpolation is turned off (get duplicate angles basiclalyt but idk why this happens. maybe future work)


use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
use csv::{ReaderBuilder, WriterBuilder};

fn main() -> Result<(), Box<dyn Error>> {
    let dir = env::args()
        .nth(1)
        .expect("Usage: csv_duplicate_fixer <directory_path>");
    let base_dir = Path::new(&dir);

    let csv_files = collect_csvs(base_dir)?;

    for path in csv_files {
        match process_csv(&path) {
            Ok(info) => println!(
                "{}: {} lines, {} records, {} fixes", 
                path.display(), info.total_lines, info.data_rows, info.fixes
            ),
            Err(e) => eprintln!("Error in {}: {}", path.display(), e),
        }
    }

    Ok(())
}

struct ProcessInfo {
    total_lines: usize,
    data_rows: usize,
    fixes: usize,
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

fn process_csv(path: &Path) -> Result<ProcessInfo, Box<dyn Error>> {
    // Read all lines
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut lines = Vec::new();
    for line in reader.lines() {
        lines.push(line?);
    }
    let total_lines = lines.len();

    // Must have at least 2 lines to have data
    if total_lines < 2 {
        return Ok(ProcessInfo { total_lines, data_rows: 0, fixes: 0 });
    }

    // Skip the first identification line
    let id_line = lines.remove(0);

    // Parse the rest as CSV (no headers)
    let blob = lines.join("\n");
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(blob.as_bytes());

    let mut data: Vec<Vec<String>> = rdr
        .records()
        .map(|r| {
            r.map(|rec| rec.iter().map(|s| s.to_string()).collect::<Vec<_>>())
             .map_err(Into::into)
        })
        .collect::<Result<_, Box<dyn Error>>>()?;

    let mut fixes = 0;
    let mut i = 0;

    while i + 2 < data.len() {
        // Detect two duplicates in column 4
        if data[i][4] == data[i + 1][4] {
            let x1_val: f64 = data[i][4].parse()?;
            let x2_index = i + 1;
            let mut dup_count = 2;

            // Count how many in a row match x1
            while i + dup_count < data.len() && data[i + dup_count][4] == data[i][4] {
                dup_count += 1;
            }
            if dup_count > 2 {
                let line_no = i + dup_count + 1; // +1 for identification line
                eprintln!(
                    "[{}] More than two consecutive duplicates (value: {}) at line {}",
                    path.display(), data[i][4], line_no
                );
                return Err(
                    format!("Too many duplicates of '{}' in {}", data[i][4], path.display()).into()
                );
            }

            // Next value Y at index i+2
            let y_index = i + 2;
            let mut y_val: f64 = data[y_index][4].parse()?;

            // Ensure Y >= X1 and X2
            if y_val < x1_val {
                // If Y is last record, set to 8192
                if y_index == data.len() - 1 {
                    y_val = 8192.0;
                    data[y_index][4] = "8192".to_string();
                } else {
                    let line_no = y_index + 1; // +1 for id line
                    return Err(
                        format!(
                            "Y value {} at line {} is less than duplicate {} in {}",
                            data[y_index][4], line_no, data[i][4], path.display()
                        )
                        .into()
                    );
                }
            }

            // Recompute average using possibly-updated Y
            let avg = ((x1_val + y_val) / 2.0).round() as i64;
            data[x2_index][4] = avg.to_string();
            fixes += 1;

            i += dup_count;
        } else {
            i += 1;
        }
    }

    // Write back: id_line + CSV rows
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(&mut writer, "{}", id_line)?;
    let mut wtr = WriterBuilder::new().has_headers(false).from_writer(writer);
    for row in &data {
        wtr.write_record(row)?;
    }
    wtr.flush()?;

    Ok(ProcessInfo { total_lines, data_rows: data.len(), fixes })
}
