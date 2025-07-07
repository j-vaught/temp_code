# grid_and_cube.py

import numpy as np
from radar_params import RadarPar

def create_iq_cube(radar: RadarPar):
    """
    Create the empty I/Q data cube for one full 360° scan.

    Parameters
    ----------
    radar : RadarPar
        Instance of the RadarPar dataclass with all radar settings.

    Returns
    -------
    cube : np.ndarray
        Empty complex64 I/Q cube of shape (n_az, n_pulses, n_rng).
    """
    # Number of range bins to cover max_range
    n_rng = int(np.ceil(radar.max_range / radar.dr))
    # Number of azimuth steps to cover 360°
    n_az  = int(np.round(360.0 / radar.az_step))
    # Allocate the cube: [azimuth, pulse index, range bin]
    cube = np.zeros((n_az, radar.n_pulses, n_rng), dtype=np.complex64)
    return cube

if __name__ == "__main__":
    # Instantiate parameters
    r = RadarPar()

    # Create the cube
    cube = create_iq_cube(r)

    # Report
    print(f"Cube shape:      {cube.shape}")            # expect (~4000, 1024, ~4000)
    mem_bytes = cube.nbytes
    print(f"Memory usage:    {mem_bytes / 1e6:.1f} MB")  # expect ~120 MB

    # Simple test to guard against runaway allocation
    assert mem_bytes < 6000e6, "I/Q cube too large (>200 MB)"
    print("Part 2: Range–azimuth grid & I/Q cube OK!")
