"""vcma_mtj: macrospin simulation of voltage-controlled magnetic anisotropy
(VCMA) switching in an elliptical perpendicular MTJ free layer."""

from .demag import DemagFactors, demag_factors, elliptic_cylinder_Nx, elliptic_cylinder_Nz
from .device import GAMMA_E, MU0, Device, deng2017
from .energy import barrier, energy_density, energy_map, equilibrium, unit_vector
from .llg import SimResult, final_state, simulate
from .pulses import Pulse, PulseTrain

__version__ = "1.0.0"

__all__ = [
    "DemagFactors", "demag_factors", "elliptic_cylinder_Nx", "elliptic_cylinder_Nz",
    "GAMMA_E", "MU0", "Device", "deng2017",
    "barrier", "energy_density", "energy_map", "equilibrium", "unit_vector",
    "SimResult", "final_state", "simulate",
    "Pulse", "PulseTrain",
]
