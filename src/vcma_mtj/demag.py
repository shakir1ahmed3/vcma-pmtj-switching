"""Demagnetization (shape-anisotropy) factors of a uniformly magnetized
elliptic cylinder.

The free layer of the Deng-type pMTJ is a thin *elliptic cylinder*
(elliptical footprint, flat top and bottom), not an ellipsoid, so the
classical Osborn/Stoner ellipsoid formulas do not apply. We use the exact
Fourier-space expressions of

    M. Beleggia, M. De Graef, Y. T. Millev, D. A. Goode and G. Rowlands,
    "Demagnetization factors for elliptic cylinders",
    J. Phys. D: Appl. Phys. 38, 3333 (2005)

as implemented in Appendix A of

    M. Beleggia, M. De Graef and Y. T. Millev,
    "The equivalent ellipsoid of a magnetized body",
    J. Phys. D: Appl. Phys. 39, 891 (2006).

Notation
--------
a, b : in-plane semi-axes (a >= b), along x and y
d    : half-thickness (t / 2), along z
beta = b / a          (in-plane aspect ratio, 0 < beta <= 1)
tau  = d / a          (half-thickness to major semi-axis)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate
from scipy.special import ellipe, ellipk, hyp2f1


def _g(phi: float, beta: float) -> float:
    return 1.0 - (1.0 - beta**2) * np.cos(phi) ** 2


def elliptic_cylinder_Nz(beta: float, tau: float) -> float:
    """Out-of-plane demagnetization factor N_z(beta, tau)."""
    m = 1.0 - beta**2  # parameter of the complete elliptic integrals (scipy convention)

    def integrand(phi: float) -> float:
        return hyp2f1(-0.5, 0.5, 2.0, -(beta**2) / (tau**2 * _g(phi, beta)))

    integral, _ = integrate.quad(integrand, 0.0, np.pi / 2, limit=200)
    return float(1.0 + 8.0 * beta / (3.0 * np.pi**2 * tau) * ellipk(m) - (2.0 / np.pi) * integral)


def elliptic_cylinder_Nx(beta: float, tau: float) -> float:
    """In-plane demagnetization factor along the major axis, N_x(beta, tau).

    Valid for beta < 1. For a circular cylinder (beta -> 1) use
    N_x = N_y = (1 - N_z) / 2.
    """
    if np.isclose(beta, 1.0):
        return 0.5 * (1.0 - elliptic_cylinder_Nz(1.0, tau))
    m = 1.0 - beta**2
    term1 = 8.0 * beta / (3.0 * np.pi**2 * tau * (1.0 - beta**2)) * (beta**2 * ellipk(m) - ellipe(m))

    def integrand(phi: float) -> float:
        g = _g(phi, beta)
        return (np.cos(phi) ** 2 / g) * hyp2f1(-0.5, 0.5, 2.0, -(beta**2) / (tau**2 * g))

    integral, _ = integrate.quad(integrand, 0.0, np.pi / 2, limit=200)
    return float(term1 + (2.0 * beta**2 / np.pi) * integral)


@dataclass(frozen=True)
class DemagFactors:
    Nx: float
    Ny: float
    Nz: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.Nx, self.Ny, self.Nz)

    def __iter__(self):
        return iter(self.as_tuple())


def demag_factors(length: float, width: float, thickness: float) -> DemagFactors:
    """Demagnetization factors of an elliptic cylinder.

    Parameters
    ----------
    length, width : full in-plane axes (2a, 2b) in metres, length >= width
    thickness     : full thickness t in metres

    Returns
    -------
    DemagFactors(Nx, Ny, Nz) with Nx + Ny + Nz = 1.
    """
    if width > length:
        raise ValueError("Use length >= width (x is the major axis).")
    a, b, d = length / 2.0, width / 2.0, thickness / 2.0
    beta, tau = b / a, d / a
    Nz = elliptic_cylinder_Nz(beta, tau)
    Nx = elliptic_cylinder_Nx(beta, tau)
    return DemagFactors(Nx=float(Nx), Ny=float(1.0 - Nx - Nz), Nz=float(Nz))
