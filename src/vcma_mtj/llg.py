"""Macrospin Landau-Lifshitz-Gilbert integrator.

    dm/dt = -gamma / (1 + alpha^2) [ m x B_eff + alpha m x (m x B_eff) ]

with B_eff = mu0 H_eff,

    H_eff(V) = 2 K_u(V) / (mu0 Ms) m_z z^  -  Ms (Nx m_x, Ny m_y, Nz m_z)  +  H_bias

(Eq. 6 of the report, without the thermal and STT terms). Integration uses a
fixed-step fourth-order Runge-Kutta scheme with renormalisation of |m| after
every step. The inner loop is compiled with numba when it is available and
falls back to pure Python otherwise (same results, ~100x slower).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .device import MU0, Device
from .pulses import PulseTrain

try:  # optional JIT
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# indices into the flat parameter vector passed to the kernel
_KUB, _KI0, _XI, _TMGO, _TFL, _MS, _NX, _NY, _NZ, _BX, _BY, _BZ, _GAM, _AW, _AR, _RD = range(16)


@njit(cache=True)
def _voltage(t, pulses):
    for k in range(pulses.shape[0]):
        t0 = pulses[k, 0]
        tr = pulses[k, 1]
        th = pulses[k, 2]
        tf = pulses[k, 3]
        amp = pulses[k, 4]
        t1 = t0 + tr
        t2 = t1 + th
        t3 = t2 + tf
        if t >= t0 and t < t1:
            return amp * (t - t0) / tr
        if t >= t1 and t < t2:
            return amp
        if t >= t2 and t < t3:
            return amp * (1.0 - (t - t2) / tf)
    return 0.0


@njit(cache=True)
def _alpha(t, pulses, p):
    a_write = p[_AW]
    a_relax = p[_AR]
    if a_relax < 0.0 or pulses.shape[0] == 0:
        return a_write
    if t < pulses[0, 0]:
        return a_write
    for k in range(pulses.shape[0]):
        t0 = pulses[k, 0]
        t_end = t0 + pulses[k, 1] + pulses[k, 2] + pulses[k, 3]
        if t >= t0 and t <= t_end + p[_RD]:
            return a_write
    return a_relax


@njit(cache=True)
def _rhs(t, mx, my, mz, pulses, p):
    V = _voltage(t, pulses)
    Ku = p[_KUB] + (p[_KI0] - p[_XI] * V / p[_TMGO]) / p[_TFL]
    mu0Ms = 4.0e-7 * np.pi * p[_MS]
    Bx = -mu0Ms * p[_NX] * mx + p[_BX]
    By = -mu0Ms * p[_NY] * my + p[_BY]
    Bz = (2.0 * Ku / p[_MS] - mu0Ms * p[_NZ]) * mz + p[_BZ]
    # m x B
    cx = my * Bz - mz * By
    cy = mz * Bx - mx * Bz
    cz = mx * By - my * Bx
    # m x (m x B)
    dx = my * cz - mz * cy
    dy = mz * cx - mx * cz
    dz = mx * cy - my * cx
    a = _alpha(t, pulses, p)
    pref = -p[_GAM] / (1.0 + a * a)
    return pref * (cx + a * dx), pref * (cy + a * dy), pref * (cz + a * dz)


@njit(cache=True)
def _integrate(m0, t_end, dt, pulses, p, store_every):
    n_steps = int(np.round(t_end / dt))
    n_out = n_steps // store_every + 1
    T = np.empty(n_out)
    M = np.empty((n_out, 3))
    mx, my, mz = m0[0], m0[1], m0[2]
    T[0] = 0.0
    M[0, 0] = mx
    M[0, 1] = my
    M[0, 2] = mz
    j = 1
    h = 0.5 * dt
    for i in range(n_steps):
        t = i * dt
        k1x, k1y, k1z = _rhs(t, mx, my, mz, pulses, p)
        k2x, k2y, k2z = _rhs(t + h, mx + h * k1x, my + h * k1y, mz + h * k1z, pulses, p)
        k3x, k3y, k3z = _rhs(t + h, mx + h * k2x, my + h * k2y, mz + h * k2z, pulses, p)
        k4x, k4y, k4z = _rhs(t + dt, mx + dt * k3x, my + dt * k3y, mz + dt * k3z, pulses, p)
        mx += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        my += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        mz += dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        nrm = np.sqrt(mx * mx + my * my + mz * mz)
        mx /= nrm
        my /= nrm
        mz /= nrm
        if (i + 1) % store_every == 0 and j < n_out:
            T[j] = (i + 1) * dt
            M[j, 0] = mx
            M[j, 1] = my
            M[j, 2] = mz
            j += 1
    return T[:j], M[:j]


@dataclass
class SimResult:
    t: np.ndarray  # s
    m: np.ndarray  # (n, 3)
    V: np.ndarray  # V
    pulses: PulseTrain
    bias: tuple[float, float, float]

    @property
    def t_ns(self) -> np.ndarray:
        return self.t * 1e9

    @property
    def mx(self) -> np.ndarray:
        return self.m[:, 0]

    @property
    def my(self) -> np.ndarray:
        return self.m[:, 1]

    @property
    def mz(self) -> np.ndarray:
        return self.m[:, 2]

    @property
    def final(self) -> np.ndarray:
        return self.m[-1]

    @property
    def switched(self) -> bool:
        return bool(np.sign(self.m[-1, 2]) != np.sign(self.m[0, 2]))

    def switching_time(self, level: float = 0.0) -> float:
        """First time (s) m_z crosses ``level`` with the sign opposite to its start
        and stays there until the end of the run. nan if it never settles."""
        s0 = np.sign(self.m[0, 2])
        same = np.sign(self.m[:, 2] - level) == s0
        if same[-1]:
            return float("nan")
        idx = np.where(same)[0]
        return float(self.t[idx[-1] + 1]) if idx.size else float(self.t[0])


def _param_vector(device: Device, bias, alpha_relax, relax_delay) -> np.ndarray:
    Bx, By, Bz = bias
    return np.array([
        device.K_U_bulk, device.K_I0, device.xi, device.t_MgO, device.t_FL, device.Ms,
        device.Nx, device.Ny, device.Nz, Bx, By, Bz, device.gamma, device.alpha,
        -1.0 if alpha_relax is None else float(alpha_relax), float(relax_delay),
    ])


def simulate(
    device: Device,
    pulses: PulseTrain,
    m0,
    t_end: float,
    bias=(0.0, 0.0, 0.0),
    dt: float = 0.1e-12,
    alpha_relax: float | None = None,
    relax_delay: float = 0.1e-9,
    store_every: int = 10,
) -> SimResult:
    """Integrate the LLG equation.

    Parameters
    ----------
    device      : Device
    pulses      : PulseTrain driving V(t)
    m0          : initial unit vector
    t_end       : total simulated time (s)
    bias        : external/bias-layer field (Bx, By, Bz) as mu0*H in tesla
    dt          : RK4 time step (s); 0.1 ps resolves all dynamics here
    alpha_relax : optional larger damping used outside the pulse windows
                  (from ``relax_delay`` after each pulse ends until the next
                  pulse starts) to shorten the post-pulse ring-down. ``None``
                  keeps ``device.alpha`` throughout.
    store_every : keep every n-th RK4 step in the returned trajectory
    """
    m0 = np.asarray(m0, dtype=float)
    m0 = m0 / np.linalg.norm(m0)
    p = _param_vector(device, bias, alpha_relax, relax_delay)
    T, M = _integrate(m0, float(t_end), float(dt), pulses.as_array(), p, int(store_every))
    return SimResult(t=T, m=M, V=pulses(T), pulses=pulses, bias=tuple(float(b) for b in bias))


def final_state(device: Device, pulses: PulseTrain, m0, t_end: float, bias=(0.0, 0.0, 0.0),
                dt: float = 0.1e-12, alpha_relax: float | None = None,
                relax_delay: float = 0.1e-9) -> np.ndarray:
    """Only the final magnetization (cheap; used for switching maps)."""
    m0 = np.asarray(m0, dtype=float)
    m0 = m0 / np.linalg.norm(m0)
    p = _param_vector(device, bias, alpha_relax, relax_delay)
    n_steps = int(np.round(t_end / dt))
    _, M = _integrate(m0, float(t_end), float(dt), pulses.as_array(), p, max(n_steps, 1))
    return M[-1]
