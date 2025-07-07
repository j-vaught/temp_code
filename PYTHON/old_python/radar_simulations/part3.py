import numpy as np
from scipy.signal import chirp, windows
from radar_params import RadarPar


def generate_chirp(
    radar: RadarPar,
    Nsamp: int = 512,           # Number of samples in pulse; change for resolution/speed
    window_beta: float = 6.0    # Kaiser window beta; adjust for sidelobe level
) -> np.ndarray:
    """
    Generate a baseband LFM chirp pulse, window it, and normalize in time-domain.

    Returns
    -------
    s : np.ndarray
        Windowed, unit-energy LFM pulse of length Nsamp.
    """
    # Pulse duration determined by bandwidth
    tau = 1.0 / radar.pulse_bw  # change to adjust pulse length
    t = np.linspace(0, tau, Nsamp, endpoint=False)

    # Create linear FM from 0 Hz up to radar.pulse_bw
    s = chirp(t, f0=0.0, t1=tau, f1=radar.pulse_bw, method='linear')

    # Apply Kaiser window to reduce sidelobes
    w = windows.kaiser(Nsamp, window_beta)
    s *= w

    # Normalize pulse energy to 1 (unit-energy); crucial for matched filter
    s /= np.linalg.norm(s)
    return s


def matched_filter(s: np.ndarray) -> np.ndarray:
    """
    Create matched-filter kernel by time-reversing and conjugating the pulse,
    then normalize main peak to unity.

    Returns
    -------
    h : np.ndarray
        Matched-filter impulse response of same length as s.
    """
    h = np.conj(s[::-1])
    # Normalize so the main tap of the impulse response is exactly 1
    h /= np.max(np.abs(h))    # change if you want a different normalization
    return h


if __name__ == "__main__":
    # Radar parameters instance
    r = RadarPar()

    # === User-adjustable settings ===
    Nsamp = 512               # Number of samples in waveform (change for speed vs. resolution)
    window_beta = 6.0         # Kaiser-beta for sidelobe control (e.g., 6→–30 dB, 8→–40 dB)
    # ================================

    # 1) Generate and check the chirp
    s = generate_chirp(r, Nsamp=Nsamp, window_beta=window_beta)
    print(f"Generated chirp length: {len(s)} samples.")

    # Energy normalization test
    energy = np.linalg.norm(s)
    print(f"Pulse energy: {energy:.6f} (should be ~1.0)")
    assert np.allclose(energy, 1.0, atol=1e-6), "Energy normalization failed"

    # 2) Create and check matched filter
    h = matched_filter(s)
    print(f"Matched filter length: {len(h)} samples.")

    # Impulse response test for PSLR
    impulse = np.zeros(Nsamp)
    impulse[0] = 1.0
    resp = np.abs(np.convolve(impulse, h, mode='full'))
    main_idx = Nsamp - 1
    main_amp = resp[main_idx]
    # Guard a two-sample gap around main peak when searching sidelobes
    sidelobes = np.concatenate((resp[:main_idx-1], resp[main_idx+2:]))
    peak_sl = np.max(sidelobes)
    pslr_db = -20 * np.log10(peak_sl / main_amp)
    print(f"Peak sidelobe level: {pslr_db:.2f} dB (expected ~ -30 dB)")

    # Tolerance for PSLR test (adjust if you tweak window_beta)
    assert -35 < pslr_db < -25, "PSLR outside expected range (-35 to -25 dB)"

    print("Part 3: Waveform & matched filter OK!")