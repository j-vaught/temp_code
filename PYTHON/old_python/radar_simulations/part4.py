import numpy as np
from radar_params import RadarPar


def generate_antenna_pattern(
    beamwidth_deg: float,
    n_points: int = 360
) -> np.ndarray:
    """
    Generate a 1-D antenna gain pattern (sinc^2) over 360 degrees.

    Parameters
    ----------
    beamwidth_deg : float
        Half-power beamwidth (3 dB) of the antenna in degrees.
    n_points : int
        Number of discrete azimuth points (default 360 for 1 deg resolution).

    Returns
    -------
    ant_pat : np.ndarray
        Array of antenna gain (linear scale, normalized) at each degree 0..359.
    """
    # Convert beamwidth to radians for sinc argument
    bw_rad = np.deg2rad(beamwidth_deg)
    # Azimuth vector centered at zero [-π, π)
    az = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    # Sinc argument scaling to control beamwidth
    sinc_arg = (2 * np.pi / bw_rad) * az
    # Compute normalized sinc(x) = sin(πx)/(πx)
    pattern = np.sinc(sinc_arg / np.pi)
    # Square of sinc gives main lobe shape
    ant_pat = pattern**2
    # Normalize to unity gain at boresight (index 0)
    ant_pat /= np.max(ant_pat)
    return ant_pat


def scan_gain_pattern(
    radar: RadarPar,
    ant_pat: np.ndarray
):
    """
    Simulate scanning the antenna and record azimuth and gain per pulse.

    Returns
    -------
    az_list : list of float
        Continuous azimuth (deg) for each pulse.
    gain_list : list of float
        Antenna gain (linear) at nearest integer degree.
    wraps : int
        Number of times azimuth wraps from near 360 back to 0.
    """
    az_list = []
    gain_list = []
    wraps = 0
    prev_az = None

    # Total pulses per 360°
    n_az = int(np.round(360.0 / radar.az_step))
    # Step through pulses +1 to detect the wrap event
    for pulse_idx in range(n_az + 1):
        # Compute azimuth for this pulse
        az = (pulse_idx * radar.az_step) % 360.0
        az_list.append(az)
        # Nearest-degree index
        idx = int(round(az)) % 360
        gain_list.append(ant_pat[idx])
        # Detect wrap: next az < previous az
        if prev_az is not None and az < prev_az:
            wraps += 1
        prev_az = az

    # Exclude the extra sample used for wrap detection
    return az_list[:-1], gain_list[:-1], wraps


if __name__ == "__main__":
    # Instantiate radar parameters
    r = RadarPar()
    # User-tunable: half-power (-3 dB) beamwidth in degrees
    beamwidth_deg = 1.5

    # Generate the antenna gain pattern
    ant_pat = generate_antenna_pattern(beamwidth_deg)

    # Simulate a full 360° scan of the rotating antenna
    az_hist, gain_hist, wraps = scan_gain_pattern(r, ant_pat)

    # Test that exactly one wrap occurs
    print(f"Azimuth wrap count: {wraps}")
    assert wraps == 1, "AZ wrap did not occur exactly once"

    # Test that number of pulses matches expectation
    expected_pulses = int(np.round(360.0 / r.az_step))
    assert len(az_hist) == expected_pulses, (
        f"Pulse count {len(az_hist)} does not match expected {expected_pulses}"
    )

    # Test gain bounds
    assert all(0.0 <= g <= 1.0 for g in gain_hist), "Gain out of [0,1] range"

    print("Part 4: Antenna pattern & scan OK!")
