import numpy as np
from dataclasses import dataclass
from scipy.constants import c
from scipy.signal import fftconvolve
from radar_params import RadarPar
from part3 import generate_chirp, matched_filter

@dataclass
class Target:
    """
    Simple target model.

    Attributes
    ----------
    x : float
        X-coordinate in meters (0° azimuth axis).
    y : float
        Y-coordinate in meters (90° azimuth axis).
    vx : float
        Velocity component along X [m/s].
    vy : float
        Velocity component along Y [m/s].
    rcs_dBsm : float
        Radar cross-section in dBsm.
    """
    x: float
    y: float
    vx: float
    vy: float
    rcs_dBsm: float


def inject_targets(
    cube: np.ndarray,
    targets: list[Target],
    radar: RadarPar
) -> None:
    """
    Add complex returns from each target into the raw I/Q cube.

    For each azimuth line and each pulse, compute range bin,
    Doppler phase, and add amplitude = linear RCS.
    """
    n_az, n_pulses, _ = cube.shape
    # Precompute matched-filter normalization later, here pure injection
    for az_idx in range(n_az):
        # boresight unit vector for this azimuth
        theta = np.deg2rad(az_idx * radar.az_step)
        cos_th, sin_th = np.cos(theta), np.sin(theta)

        for pulse_idx in range(n_pulses):
            t = pulse_idx / radar.prf
            for tgt in targets:
                # Slant range (flat Earth)
                r_xy = np.hypot(tgt.x, tgt.y)
                # Map to range bin
                bin_idx = int(round(r_xy / radar.dr))
                # Doppler frequency for radial motion
                fd = 2 * (tgt.vx * cos_th + tgt.vy * sin_th) / radar.wavelength
                phase = np.exp(1j * 2 * np.pi * fd * t)
                # Linear RCS power (omit propagation losses for test)
                P_rcs = 10 ** (tgt.rcs_dBsm / 10)
                # Inject return
                cube[az_idx, pulse_idx, bin_idx] += P_rcs * phase


def test_static_target():
    # Instantiate radar and waveform
    r = RadarPar()
    Nsamp = 512
    beta = 6.0
    s = generate_chirp(r, Nsamp, beta)
    h = matched_filter(s)

    # Build raw I/Q cube at least one full revolution
    n_az = int(np.round(360.0 / r.az_step))
    n_rng = int(np.ceil(r.max_range / r.dr))
    cube = np.zeros((n_az, r.n_pulses, n_rng), dtype=np.complex64)

    # Single stationary target at (x=3000 m, y=0)
    tgt = Target(x=3000, y=0, vx=0, vy=0, rcs_dBsm=20)
    inject_targets(cube, [tgt], r)

    # Pulse-compress along range axis
    compressed = fftconvolve(cube, h[None, None, :], mode='same')

    # Find the azimuth index where target is illuminated (az=0 => idx=0)
    az_idx = 0
    bin_idx = int(round(3000 / r.dr))
    # Extract across pulses then find peak magnitude
    mag = np.abs(compressed[az_idx, :, bin_idx])
    peak_db = 20 * np.log10(np.max(mag))
    print(f"Static 20 dBsm at 3 km → peak = {peak_db:.2f} dB (expect ~40 dB)")
    assert 38 < peak_db < 42, "Target peak out of expected 38–42 dB"
    print("Part 5: Target synthesiser OK!")

if __name__ == "__main__":
    test_static_target()
