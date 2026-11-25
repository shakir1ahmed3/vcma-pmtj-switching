"""The simulation scenarios reported in the write-up, in one place.

Every figure in ``figures/`` and in the report is produced from the
definitions below (see ``scripts/make_figures.py``). Voltages for the
"further evaluations" are chosen relative to the critical voltages of the
corrected device (V_xz = 4.74 V, V_yz = 7.81 V):

* near-critical, low field : 4.8 V  (just above V_xz),   Bx = 5 mT
* high voltage, low field  : 8.5 V  (above V_yz),        Bx = 5 mT
* high field               : 0 / 1.6 / 3 / 5 V,          Bx = 50 mT  (V_crit = 2.61 V)
* VCMA-assisted field switching : 4.8 V, B = (0.5, 0, -5) mT

Note: 2 V across a 2 nm MgO barrier is already ~1 V/nm, close to typical
dielectric breakdown. The higher voltages are used to map the macrospin
regimes, not as proposed operating points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .device import Device, deng2017
from .energy import equilibrium
from .llg import SimResult, final_state, simulate
from .pulses import Pulse, PulseTrain

T_START = 0.2e-9  # pulse start used in the "further evaluations"
POST_PULSE = 7.0e-9  # relaxation window simulated after the last pulse
ALPHA_RELAX = 0.10  # damping used between/after pulses to shorten ring-down


# ----------------------------------------------------------------------
# A. Reproduction of Deng, Liang & Gupta (2017), Fig. 2
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DengCase:
    V0: float = 1.6
    Bx: float = 50e-3
    t_on: tuple[float, ...] = (0.35e-9, 0.70e-9, 1.05e-9)
    t_end: float = 10e-9
    m0: tuple[float, float, float] = (0.0, 0.0, -1.0)  # idealised pole, as in the paper's figure


def run_deng(device: Device | None = None, case: DengCase = DengCase(), m0=None) -> dict[float, SimResult]:
    device = device or deng2017()
    m0 = case.m0 if m0 is None else m0
    return {
        t_on: simulate(device, PulseTrain(Pulse.from_t_on(case.V0, t_on)), m0, case.t_end,
                       bias=(case.Bx, 0.0, 0.0), store_every=5)
        for t_on in case.t_on
    }


# ----------------------------------------------------------------------
# D. Near-critical voltage, low field
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class NearCritical:
    V0: float = 4.8
    Bx: float = 5e-3
    t_hold_long: float = 9e-9  # reveals the precession period (~2 ns)
    t_hold_half: float = 1.0e-9  # ~T/2: centre of the first switching window
    t_hold_quarter: float = 0.5e-9  # ~T/4: first m_z zero crossing, window edge
    gap: float = 5e-9  # off-time between the two pulses of the double-pulse test


# ----------------------------------------------------------------------
# E. High voltage, low field
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class HighVoltage:
    V0: float = 8.5
    Bx: float = 5e-3
    t_hold_case1: float = 25e-9  # pulse left on until m orbits the in-plane x axis
    t_hold_case2: float = 18.4e-9  # pulse removed before the first m_z zero crossing


# ----------------------------------------------------------------------
# F. High field
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class HighField:
    Bx: float = 50e-3
    voltages: tuple[float, ...] = (0.0, 1.6, 3.0, 5.0)
    t_hold: float = 3e-9


# ----------------------------------------------------------------------
# G. VCMA-assisted field-driven switching
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FieldAssisted:
    V0: float = 4.8
    bias: tuple[float, float, float] = (0.5e-3, 0.0, -5e-3)
    t_hold: float = 15e-9
    gap: float = 5e-9


LANDSCAPE_VOLTAGES = (0.0, 2.5, 4.75, 8.5)  # zero field: strong PMA, weakened, ~V_xz, > V_yz


def run_pulse(device: Device, V0: float, t_hold: float, bias, m0=None, double_gap: float | None = None,
              post: float = POST_PULSE, alpha_relax: float | None = ALPHA_RELAX,
              store_every: int = 10) -> SimResult:
    """Single (or double, if ``double_gap`` is given) pulse starting at T_START
    from the +z equilibrium, followed by ``post`` seconds of relaxation."""
    if m0 is None:
        m0 = equilibrium(device, 0.0, bias, +1)
    if double_gap is None:
        pulses = PulseTrain.single(V0, t_hold, t_start=T_START)
    else:
        pulses = PulseTrain.double(V0, t_hold, gap=double_gap, t_start=T_START)
    return simulate(device, pulses, m0, pulses.t_end + post, bias=bias,
                    alpha_relax=alpha_relax, store_every=store_every)


def switching_vs_width(device: Device, V0: float, t_holds, bias, m0=None,
                       alpha_relax: float | None = ALPHA_RELAX) -> np.ndarray:
    """Final m_z after a single pulse, for each flat-top duration in ``t_holds``."""
    if m0 is None:
        m0 = equilibrium(device, 0.0, bias, +1)
    out = np.empty(len(t_holds))
    for i, th in enumerate(t_holds):
        pulses = PulseTrain.single(V0, th, t_start=T_START)
        out[i] = final_state(device, pulses, m0, pulses.t_end + POST_PULSE, bias=bias,
                             alpha_relax=alpha_relax)[2]
    return out


def switching_map(device: Device, voltages, t_ons, bias, m0, t_settle: float = 8e-9) -> np.ndarray:
    """Final m_z on a (voltage, ON-time) grid for Deng-style pulses starting at t = 0
    (constant damping, no relaxation boost). Shape (len(voltages), len(t_ons))."""
    out = np.empty((len(voltages), len(t_ons)))
    for i, V in enumerate(voltages):
        for j, t_on in enumerate(t_ons):
            pulses = PulseTrain(Pulse.from_t_on(V, t_on))
            out[i, j] = final_state(device, pulses, m0, t_on + t_settle, bias=bias)[2]
    return out


def measured_period(res: SimResult, t_from: float, t_to: float) -> float:
    """Mean spacing (s) between successive m_z minima inside [t_from, t_to]."""
    from scipy.signal import find_peaks

    sel = (res.t >= t_from) & (res.t <= t_to)
    pk, _ = find_peaks(-res.mz[sel], prominence=0.05)
    if len(pk) < 2:
        return float("nan")
    return float(np.mean(np.diff(res.t[sel][pk])))


def windows_from_scan(t_holds, mz_final) -> list[tuple[float, float]]:
    """Contiguous ranges of t_hold where the layer switched (m_z < 0)."""
    sw = np.asarray(mz_final) < 0
    out, start = [], None
    for th, s in zip(t_holds, sw):
        if s and start is None:
            start = th
        if not s and start is not None:
            out.append((start, prev))
            start = None
        prev = th
    if start is not None:
        out.append((start, prev))
    return out


def window_period(windows) -> float:
    """Effective period (s) implied by switching windows: least-squares fit of the
    window centres to (n + 1/2) T, n = 0, 1, 2, ...

    This is the period that sets the (n+1/4)T < t_hold < (n+3/4)T rule. It differs
    slightly from ``measured_period`` on a single long pulse (pulse ramps, and the
    precession period drifts as the cone angle changes), so use it, not that one,
    when overlaying the rule on a t_hold scan."""
    if not windows:
        return float("nan")
    c = np.array([(a + b) / 2 for a, b in windows])
    n = np.arange(len(c)) + 0.5
    return float(c @ n / (n @ n))
