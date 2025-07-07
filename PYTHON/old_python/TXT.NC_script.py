#!/usr/bin/env python3
import zipfile
import numpy as np
from netCDF4 import Dataset
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# 1) Open the ZIP ONCE and keep it open while processing all TXT files
# ─────────────────────────────────────────────────────────────────────────────
zip_path = "20250411_122849.zip"
with zipfile.ZipFile(zip_path, "r") as z:
    # Gather & sort all TXT names
    txt_files = sorted([f for f in z.namelist() if f.endswith(".txt")])

    N_time  = len(txt_files)    # Number of sweeps (≈270)
    N_angle = 2048              # Expect 0–2047 angle indices per sweep
    N_range = 1024              # Expect 1024 range bins per sweep

    # ─────────────────────────────────────────────────────────────────────────
    # 2) Preallocate a 3D array [time, angle, range] of int8 (0–15).
    #    Any missing angles will stay as zeros.
    # ─────────────────────────────────────────────────────────────────────────
    all_echo = np.zeros((N_time, N_angle, N_range), dtype=np.int8)

    # We will also collect a list of ISO‐formatted time strings for each sweep
    time_strings = []

    # ─────────────────────────────────────────────────────────────────────────
    # 3) Loop over each TXT, parse its timestamp (local Columbia, SC),
    #    then read echoes into the correct [angle, range] slice.
    # ─────────────────────────────────────────────────────────────────────────
    for idx, fname in enumerate(txt_files):
        # Extract just the base filename, e.g.
        #   "radar_raw_data_20250411_122851.400_0.txt"
        base = fname.split("/")[-1]
        parts = base.split("_")
        # Now parts should be:
        #   0: "radar"
        #   1: "raw"
        #   2: "data"
        #   3: "20250411"
        #   4: "122851.400"
        #   5: "0.txt"
        date_str = parts[3]       # e.g. "20250411"
        time_str = parts[4]       # e.g. "122851.400"

        # Parse into a Python datetime (naive = local). The ".400" is milliseconds.
        # Format string: "%Y%m%d%H%M%S.%f"
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S.%f")

        # Build an ISO‐style string with milliseconds (local time)
        iso_time = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]  # e.g. "2025-04-11T12:28:51.400"
        time_strings.append(iso_time)

        # Read the TXT sweep itself (while ZIP is still open)
        with z.open(fname) as f:
            # Start with zeros—if an angle is missing entirely, it remains zeros
            sweep = np.zeros((N_angle, N_range), dtype=np.int8)

            for line in f:
                toks = line.decode("utf-8").strip().split()
                # Expect at least 9 tokens before the 1024 echo values
                if len(toks) < 9:
                    continue

                angle_idx = int(toks[5])                     # 0–2047
                echoes = np.array(list(map(int, toks[9:])), dtype=np.int8)

                # If fewer than 1024 echo values, zero-pad
                if echoes.size < N_range:
                    padded = np.zeros(N_range, dtype=np.int8)
                    padded[: echoes.size] = echoes
                    echoes = padded

                # Store into our 2D sweep array
                sweep[angle_idx, :] = echoes

            # Write this sweep into [time=idx, :, :]
            all_echo[idx, :, :] = sweep

# ─────────────────────────────────────────────────────────────────────────────
# 4) Now that the ZIP is closed, create the NetCDF file and define dimensions + vars
# ─────────────────────────────────────────────────────────────────────────────
nc_out = "radar_all_sweeps.nc"
nc = Dataset(nc_out, "w", format="NETCDF4")

# Define dimensions
nc.createDimension("time", N_time)
nc.createDimension("angle", N_angle)
nc.createDimension("range", N_range)

# To store time as an ISO8601 string per sweep, add a "str_len" dim
max_len = max(len(ts) for ts in time_strings)
nc.createDimension("str_len", max_len)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Create coordinate variables
# ─────────────────────────────────────────────────────────────────────────────
# time: stored as (time, str_len) of single‐character strings
time_var = nc.createVariable("time", "S1", ("time", "str_len"))
time_var.long_name = "Sweep timestamp (local, Columbia SC)"
time_var.units     = "ISO8601 string (YYYY-MM-DDTHH:MM:SS.mmm)"

# angle: a float array from 0° up to just under 360° (2048 steps)
angle_var = nc.createVariable("angle", "f4", ("angle",))
angle_var.units          = "degrees"
angle_var.long_name      = "Azimuth angle of each beam"
angle_var.standard_name  = "beam_azimuth_angle"
angle_var[:] = np.linspace(0.0, 360.0, N_angle, endpoint=False)

# range: an integer index 0–1023 (raw bin number)
range_var = nc.createVariable("range", "i4", ("range",))
range_var.units     = "bin_index"
range_var.long_name = "Range gate index"
range_var[:] = np.arange(N_range, dtype=np.int32)

# the main data variable: echo_strength[time, angle, range]
# choose a 1-byte signed int (i1) to hold 0–15; use fill_value=-1 if missing
echo_var = nc.createVariable(
    "echo_strength",
    "i1",
    ("time", "angle", "range"),
    fill_value=-1,
    zlib=True,
    complevel=4,
    shuffle=True
)
echo_var.long_name   = "Radar echo intensity (raw 4-bit counts)"
echo_var.valid_min   = 0
echo_var.valid_max   = 15
echo_var.units       = "raw count"
echo_var.coordinates = "time angle range"

# ─────────────────────────────────────────────────────────────────────────────
# 6) Write the coordinate arrays
# ─────────────────────────────────────────────────────────────────────────────
# Fill time_var as fixed-length strings: each row is one ISO string, padded with spaces
for i, ts in enumerate(time_strings):
    padded = ts.ljust(max_len)
    # Convert to array of single‐character bytes
    time_var[i, :] = np.array(list(padded.encode("utf-8")), dtype="S1")

# angle_var and range_var were already assigned above

# ─────────────────────────────────────────────────────────────────────────────
# 7) Write the full echo data
# ─────────────────────────────────────────────────────────────────────────────
echo_var[:, :, :] = all_echo

# ─────────────────────────────────────────────────────────────────────────────
# 8) Add global (file-level) attributes
# ─────────────────────────────────────────────────────────────────────────────
nc.title       = "Radar Raw‐Data Collection, 2025-04-11 12:28–12:36 UTC"
nc.institution = "University of South Carolina, iMSEL"
nc.source      = "4-bit echo counts per bin, sweeps from X-band simrad Radar"
nc.history     = "Converted from text with TXT.NC_script v1.0"
nc.references  = "none"
nc.comment     = "Each sweep has 2048 bearings, 1024 range bins, 4‐bit echo values."

# Close the NetCDF
nc.close()

print(f"Successfully created NetCDF → {nc_out}")
