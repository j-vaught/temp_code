import os
import glob
import csv
import numpy as np
from datetime import datetime, timedelta, timezone
from netCDF4 import Dataset
from multiprocessing import Pool
from contextlib import contextmanager
from time import perf_counter

# Constants
MAX_PULSE = 8192
GATE      = 868
FILL_F32  = -9999.0

@contextmanager
def timeit(label):
    t0 = perf_counter()
    yield
    print(f"{label:<20}{perf_counter() - t0:.3f}s")

def parse_info(path):
    info = {}
    with open(path, 'r') as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith('#'):
                continue
            section, key, val = [p.strip() for p in t.split(',')]
            info.setdefault(section, {})[key] = val
    return info

def list_sweeps(base_dir):
    return sorted(glob.glob(os.path.join(base_dir, 'good', '*.csv')))

def parse_csv_record(row):
    sc, rc, gn = map(int, row[1:4])
    azi = int(row[4]) / 8192 * 360.0
    echos = np.array(row[5:5+GATE], dtype=np.float32)
    return sc, rc, gn, azi, echos

def fill_buffers_from_csv(csv_path):
    echo_buf  = np.full((MAX_PULSE, GATE), FILL_F32, np.float32)
    angle_buf = np.full((MAX_PULSE,),       FILL_F32, np.float32)
    with open(csv_path, 'r') as f:
        rdr = csv.reader(f)
        next(rdr, None)
        scale = range_ = gain = None
        for p, row in enumerate(rdr):
            if p >= MAX_PULSE:
                break
            sc, rc, gn, azi, echos = parse_csv_record(row)
            if scale is None:
                scale, range_, gain = sc, rc, gn
            angle_buf[p]   = azi
            echo_buf[p, :] = echos
    return scale or 0, range_ or 0, gain or 0, echo_buf, angle_buf

def parse_filename_to_secs(fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    date_s, time_s, ms_s = stem.split('_')
    dt = datetime.strptime(date_s + time_s, '%Y%m%d%H%M%S')
    dt += timedelta(milliseconds=int(ms_s))
    return dt.replace(tzinfo=timezone.utc).timestamp()

def process_sweep(idx_path):
    idx, path = idx_path
    t0 = parse_filename_to_secs(path)
    sc, ra, gn, echo, angle = fill_buffers_from_csv(path)
    return idx, t0, sc, ra, gn, echo, angle

def main():
    base_dir    = '/Users/jacobvaught/Downloads/Research_radar_DATA/data/data_1'
    output_path = os.path.join(base_dir, 'output.nc')
    try: os.remove(output_path)
    except FileNotFoundError: pass

    with timeit('list_sweeps'):
        sweeps = list_sweeps(base_dir)
    n = len(sweeps)

    with timeit('parse_csv_parallel'):
        with Pool() as pool:
            results = pool.map(process_sweep, enumerate(sweeps))
    results.sort(key=lambda x: x[0])

    # Allocate batched arrays
    times_arr = np.empty(n,                  np.float64)
    echo_arr  = np.full((n, MAX_PULSE, GATE), FILL_F32, np.float32)
    angle_arr = np.full((n, MAX_PULSE),        FILL_F32, np.float32)
    scale_arr = np.empty(n,                  np.int32)
    range_arr = np.empty(n,                  np.int32)
    gain_arr  = np.empty(n,                  np.int32)

    for idx, t0, sc, ra, gn, echo, angle in results:
        times_arr[idx]  = t0
        echo_arr[idx]   = echo
        angle_arr[idx]  = angle
        scale_arr[idx]  = sc
        range_arr[idx]  = ra
        gain_arr[idx]   = gn

    with timeit('create_netcdf'):
        # Try parallel I/O first
        try:
            from mpi4py import MPI
            nc = Dataset(
                output_path, 'w', format='NETCDF4',
                parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info()
            )
        except Exception:
            nc = Dataset(output_path, 'w', format='NETCDF4')

        # dimensions
        nc.createDimension('time',  None)
        nc.createDimension('pulse', MAX_PULSE)
        nc.createDimension('gate',  GATE)

        # variables with chunksizes
        times_var = nc.createVariable('time',  'f8',    ('time',))

        echo_var  = nc.createVariable(
            'echo', 'f4', ('time', 'pulse', 'gate'),
            compression='zstd',             # use plugin filter
            compression_opts=19,            # ≈ zlib-9 ratio
            shuffle=True,                   # keep byte-shuffle
            chunksizes=(10, MAX_PULSE, GATE)
        )

        angle_var = nc.createVariable(
            'angle', 'f4', ('time', 'pulse'),
            compression='zstd',
            compression_opts=19,
            shuffle=True,
            chunksizes=(10, MAX_PULSE)
        )


        scale_var = nc.createVariable('scale','i4',('time',))
        range_var = nc.createVariable('range','i4',('time',))
        gain_var  = nc.createVariable('gain', 'i4',('time',))

    with timeit('write_netcdf'):
        times_var[:]  = times_arr
        echo_var[:]   = echo_arr
        angle_var[:]  = angle_arr
        scale_var[:]  = scale_arr
        range_var[:]  = range_arr
        gain_var[:]   = gain_arr

    # Optional attributes
    info_txt = os.path.join(base_dir, 'info.txt')
    if os.path.exists(info_txt):
        with timeit('write_attrs'):
            for section, kv in parse_info(info_txt).items():
                for k, v in kv.items():
                    nc.setncattr(f"{section}_{k}", v)

    nc.close()
    print(f"Finished OK, wrote {n} sweeps → {output_path}")

if __name__ == '__main__':
    main()
