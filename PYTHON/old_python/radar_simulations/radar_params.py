import numpy as np
from dataclasses import dataclass
from scipy.constants import c

@dataclass
class RadarPar:
    """
    Basic X-band rotating radar parameters.
    """
    # Hardware / waveform
    f0: float = 9.41e9           # Carrier frequency [Hz]
    pulse_bw: float = 10e6       # Pulse bandwidth [Hz]
    prf: float = 3e3             # Pulse repetition frequency [Hz]
    rotation_rate: float = 48    # Antenna rotation speed [rpm]
    tx_power: float = 25e3       # Transmit peak power [W]
    g_tx: float = 32             # Transmit gain [dBi]
    g_rx: float = 32             # Receive gain [dBi]
    n_pulses: int = 512          # Pulses per coherent processing interval (CPI)
    max_range: float = 5e3      # Maximum range to simulate [m]


    @property
    def wavelength(self) -> float:
        """Radar wavelength [m]."""
        return c / self.f0

    @property
    def dr(self) -> float:
        """
        Range-bin size [m], determined by pulse bandwidth:
            dr = c / (2 * BW)
        """
        return c / (2 * self.pulse_bw)

    @property
    def az_step(self) -> float:
        """
        Azimuth step per pulse [deg]:
            az_step = 360° * (rotation_rate [rev/min]) / (60 * PRF [pulses/s])
        """
        return 360.0 * self.rotation_rate / (60.0 * self.prf)


if __name__ == "__main__":
    # Quick sanity check
    r = RadarPar()
    print(f"Wavelength λ:     {r.wavelength:.3f} m")
    print(f"Range bin size:   {r.dr:.3f} m")
    print(f"Azimuth step:     {r.az_step:.3f}°")

    # Quick assertions
    assert 0.03 < r.wavelength < 0.04     # ≈ 0.032 m for X-band
    assert 7.0 < r.dr < 8.0               # ≈ 7.5 m range resolution
    assert 0.08 < r.az_step < 0.10        # ≈ 0.09° per pulse at 24 rpm, 1.5 kHz PRF
    print("Part 1 parameters OK!")
