"""Device and material parameters of the Deng-type elliptical pMTJ free layer,
and the closed-form (static) anisotropy quantities derived from them.

Default values follow J. Deng, G. Liang and G. Gupta, "Ultrafast and
low-energy switching in voltage-controlled elliptical pMTJ",
Sci. Rep. 7, 16562 (2017), Methods section.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .demag import demag_factors

MU0 = 4.0e-7 * np.pi  # vacuum permeability, H/m
GAMMA_E = 1.76085963e11  # electron gyromagnetic ratio, rad/(s*T)


@dataclass(frozen=True)
class Device:
    """Macrospin free-layer model.

    All quantities in SI units. If ``N`` is not given, the demagnetization
    factors are computed from the elliptic-cylinder geometry
    (``length`` x ``width`` x ``t_FL``) with the exact Beleggia expressions.
    """

    # geometry
    length: float = 150e-9  # major axis (x), m
    width: float = 50e-9  # minor axis (y), m
    t_FL: float = 1.7e-9  # free-layer thickness, m
    t_MgO: float = 2.0e-9  # MgO barrier thickness, m

    # magnetic material
    Ms: float = 1.257e6  # saturation magnetization, A/m
    K_U_bulk: float = 2.245e5  # bulk anisotropy density, J/m^3
    K_I0: float = 1.286e-3  # zero-bias interfacial anisotropy, J/m^2
    xi: float = 50e-15  # VCMA coefficient, J/(V*m)
    alpha: float = 0.01  # Gilbert damping
    gamma: float = GAMMA_E  # gyromagnetic ratio, rad/(s*T)

    N: tuple[float, float, float] | None = field(default=None)

    def __post_init__(self):
        if self.N is None:
            object.__setattr__(self, "N", demag_factors(self.length, self.width, self.t_FL).as_tuple())
        if not np.isclose(sum(self.N), 1.0, atol=1e-9):
            raise ValueError(f"Nx + Ny + Nz must equal 1, got {sum(self.N)}")

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    @property
    def Nx(self) -> float:
        return self.N[0]

    @property
    def Ny(self) -> float:
        return self.N[1]

    @property
    def Nz(self) -> float:
        return self.N[2]

    def with_(self, **changes) -> "Device":
        """Return a copy with some parameters changed."""
        return replace(self, **changes)

    # ------------------------------------------------------------------
    # anisotropy energies (J/m^3)
    # ------------------------------------------------------------------
    @property
    def Kd(self) -> float:
        """Shape-anisotropy energy scale mu0 Ms^2 / 2."""
        return 0.5 * MU0 * self.Ms**2

    @property
    def dKdV(self) -> float:
        """VCMA slope dK_u/dV = -xi / (t_MgO t_FL)  (J/m^3 per volt)."""
        return -self.xi / (self.t_MgO * self.t_FL)

    def Ku(self, V):
        """Total uniaxial anisotropy density K_u(V) = K_U,bulk + K_i(V)/t_FL."""
        K_i = self.K_I0 - self.xi * np.asarray(V) / self.t_MgO
        return self.K_U_bulk + K_i / self.t_FL

    def Keff(self, V):
        """Effective PMA density K_u(V) - K_d N_z (Eq. 4)."""
        return self.Ku(V) - self.Kd * self.Nz

    def Keff_xz(self, V):
        """Effective anisotropy for rotation in the x-z plane."""
        return self.Ku(V) - self.Kd * (self.Nz - self.Nx)

    def Keff_yz(self, V):
        """Effective anisotropy for rotation in the y-z plane."""
        return self.Ku(V) - self.Kd * (self.Nz - self.Ny)

    # ------------------------------------------------------------------
    # anisotropy fields, expressed as mu0*H in tesla
    # ------------------------------------------------------------------
    def BK(self, V):
        """mu0 H_K,u(V) = 2 K_u(V)/Ms."""
        return 2.0 * self.Ku(V) / self.Ms

    def BKeff(self, V):
        return 2.0 * self.Keff(V) / self.Ms

    def BKeff_xz(self, V):
        return 2.0 * self.Keff_xz(V) / self.Ms

    def BKeff_yz(self, V):
        return 2.0 * self.Keff_yz(V) / self.Ms

    # ------------------------------------------------------------------
    # critical voltages
    # ------------------------------------------------------------------
    def _zero_crossing(self, K0: float) -> float:
        return -K0 / self.dKdV

    @property
    def V_Ku0(self) -> float:
        """Voltage at which K_u itself vanishes."""
        return self._zero_crossing(float(self.Ku(0.0)))

    @property
    def V_c(self) -> float:
        """Voltage at which K_eff = K_u - K_d N_z vanishes."""
        return self._zero_crossing(float(self.Keff(0.0)))

    @property
    def V_xz(self) -> float:
        """Voltage at which the x-z barrier vanishes (H_K,eff,xz = 0)."""
        return self._zero_crossing(float(self.Keff_xz(0.0)))

    @property
    def V_yz(self) -> float:
        """Voltage at which the y-z barrier vanishes (H_K,eff,yz = 0)."""
        return self._zero_crossing(float(self.Keff_yz(0.0)))

    def V_crit_for_bias(self, Bx: float) -> float:
        """Smallest voltage at which an in-plane field Bx (tesla) alone removes
        the barrier between the two out-of-plane states, i.e.
        mu0 H_K,eff,xz(V) = Bx."""
        return self._zero_crossing(float(self.Keff_xz(0.0)) - 0.5 * self.Ms * Bx)

    def B_threshold_x(self, V):
        """Threshold in-plane field (tesla) that flattens the x-z barrier (Eq. 13).
        Zero once the barrier has already vanished."""
        return np.maximum(0.0, self.BKeff_xz(V))

    def B_switch_z(self, V):
        """Minimum easy-axis (z) field (tesla) for field-driven switching (Eq. 14):
        2K_u(V)/Ms - mu0 Ms [N_z - min(N_x, N_y)]."""
        return 2.0 * self.Ku(V) / self.Ms - MU0 * self.Ms * (self.Nz - min(self.Nx, self.Ny))

    # ------------------------------------------------------------------
    # precession period estimates
    # ------------------------------------------------------------------
    def period_bias_only(self, Bx: float) -> float:
        """Larmor period about a pure in-plane field, Eq. (8):
        T = 2 pi (1 + alpha^2) / (gamma mu0 H_ip)."""
        return 2.0 * np.pi * (1.0 + self.alpha**2) / (self.gamma * Bx)

    def period_kittel(self, V: float, Bx: float) -> float:
        """Kittel biaxial period for small precession about +x, Eq. (7).
        Valid only when +x is a stable state (Bx > mu0 H_K,eff,xz(V)); returns
        nan otherwise."""
        h1 = Bx - float(self.BKeff_xz(V))
        h2 = Bx + MU0 * self.Ms * (self.Ny - self.Nx)
        if h1 <= 0 or h2 <= 0:
            return float("nan")
        return 2.0 * np.pi * (1.0 + self.alpha**2) / (self.gamma * np.sqrt(h1 * h2))

    def summary(self) -> dict:
        return {
            "Nx": self.Nx,
            "Ny": self.Ny,
            "Nz": self.Nz,
            "Ku(0) [J/m^3]": float(self.Ku(0.0)),
            "Keff(0) [J/m^3]": float(self.Keff(0.0)),
            "mu0 H_K,eff(0) [mT]": 1e3 * float(self.BKeff(0.0)),
            "mu0 H_K,eff,xz(0) [mT]": 1e3 * float(self.BKeff_xz(0.0)),
            "mu0 H_K,eff,yz(0) [mT]": 1e3 * float(self.BKeff_yz(0.0)),
            "V_c (Keff=0) [V]": self.V_c,
            "V_xz [V]": self.V_xz,
            "V_yz [V]": self.V_yz,
            "V_Ku=0 [V]": self.V_Ku0,
        }


def deng2017() -> Device:
    """The Deng, Liang & Gupta (2017) free layer: 150 x 50 x 1.7 nm^3 ellipse."""
    return Device()
