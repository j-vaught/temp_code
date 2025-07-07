use crate::lib::{parse_info, list_sweeps, parse_filename_to_secs, parse_csv_record};
…
fn main() -> Result<()> {
    let args = Args::parse();
    let meta = parse_info(&args.base_dir.join("info.txt"))?.get("Metadata")…?;
    let tz   = DateTime::parse_from_rfc3339(&meta["timestamp"])?.offset().clone();

    let sweeps = list_sweeps(&args.base_dir)?;
    let mut nc = netcdf::create(&args.output)?;
    // … set dims, vars, attributes …

    for (i,path) in sweeps.iter().enumerate() {
        let secs = parse_filename_to_secs(path.file_name().unwrap().to_str().unwrap(), tz)?;
        time_var.put_values(&[secs], (i,))?;
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
        for (r, rec) in rdr.records().enumerate() {
            let (sc, rc, gn, azi, echos) = parse_csv_record(&rec?)?;
            // write into local buffers, then at end:
            inten.put_values(&ibuf, (i, .., ..))?;
            // … etc …
        }
    }
    Ok(())
}
