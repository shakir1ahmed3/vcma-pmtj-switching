"""Macrospin free-energy density and its equilibria (Eq. 3 of the report).

    E/V = -K_u(V) m_z^2 + K_d (N_x m_x^2 + N_y m_y^2 + N_z m_z^2) - Ms B_bias . m
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .device import Device


def unit_vector(theta, phi):
    """m(theta, phi) with theta the polar angle from +z."""
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


def energy_density(device: Device, m, V=0.0, bias=(0.0, 0.0, 0.0)):
    """Energy density (J/m^3) for unit vector(s) m (last axis = xyz)."""
    m = np.asarray(m, dtype=float)
    mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
    Bx, By, Bz = bias
    E_ani = -device.Ku(V) * mz**2
    E_dem = device.Kd * (device.Nx * mx**2 + device.Ny * my**2 + device.Nz * mz**2)
    E_zee = -device.Ms * (Bx * mx + By * my + Bz * mz)
    return E_ani + E_dem + E_zee


def energy_map(device: Device, theta, phi, V=0.0, bias=(0.0, 0.0, 0.0)):
    """Energy density on a (theta, phi) grid built with np.meshgrid."""
    return energy_density(device, unit_vector(theta, phi), V, bias)


def equilibrium(device: Device, V=0.0, bias=(0.0, 0.0, 0.0), hemisphere: int = +1) -> np.ndarray:
    """Local energy minimum in the upper (+1) or lower (-1) hemisphere.

    The in-plane bias cants the states away from (0, 0, +/-1), so the minima
    are found numerically. Several starting guesses are tried and the lowest
    minimum that stays in the requested hemisphere is returned.
    """

    def f(x):
        return float(energy_density(device, unit_vector(x[0], x[1]), V, bias))

    best = None
    theta0s = [0.05, 0.3, 0.6, 1.0] if hemisphere > 0 else [np.pi - 0.05, np.pi - 0.3, np.pi - 0.6, np.pi - 1.0]
    for th0 in theta0s:
        for ph0 in (0.0, np.pi / 2, np.pi, -np.pi / 2):
            res = minimize(f, x0=[th0, ph0], method="Nelder-Mead",
                           options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
            m = unit_vector(*res.x)
            if np.sign(m[2]) != np.sign(hemisphere) and abs(m[2]) > 1e-6:
                continue
            if best is None or res.fun < best[0] - 1e-9:
                best = (res.fun, m)
    if best is None:  # no minimum in that hemisphere (e.g. barrier removed)
        res = minimize(f, x0=[np.pi / 2, 0.0], method="Nelder-Mead")
        return unit_vector(*res.x)
    m = best[1]
    m[np.abs(m) < 1e-9] = 0.0
    return m


def barrier(device: Device, V=0.0, bias=(0.0, 0.0, 0.0)) -> float:
    """Energy barrier (J/m^3) between the +z state and the lowest in-plane
    saddle, estimated as min over the equator minus E(+z equilibrium)."""
    phi = np.linspace(-np.pi, np.pi, 3601)
    E_eq = energy_density(device, unit_vector(np.full_like(phi, np.pi / 2), phi), V, bias).min()
    return float(E_eq - energy_density(device, equilibrium(device, V, bias, +1), V, bias))
