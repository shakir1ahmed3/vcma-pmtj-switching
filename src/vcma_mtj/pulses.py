"""Trapezoidal voltage pulses and pulse trains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pulse:
    """Single trapezoidal pulse.

    The ON-time used by Deng et al. is ``t_rise + t_hold + t_fall``.
    """

    amplitude: float  # V
    t_start: float = 0.0  # s, beginning of the rising edge
    t_hold: float = 0.25e-9  # s, flat top
    t_rise: float = 50e-12  # s
    t_fall: float = 50e-12  # s

    @property
    def t_on(self) -> float:
        return self.t_rise + self.t_hold + self.t_fall

    @property
    def t_end(self) -> float:
        return self.t_start + self.t_on

    @classmethod
    def from_t_on(cls, amplitude: float, t_on: float, t_start: float = 0.0,
                  t_rise: float = 50e-12, t_fall: float = 50e-12) -> "Pulse":
        return cls(amplitude, t_start, t_on - t_rise - t_fall, t_rise, t_fall)

    def as_row(self) -> list[float]:
        return [self.t_start, self.t_rise, self.t_hold, self.t_fall, self.amplitude]


class PulseTrain:
    """A sequence of non-overlapping trapezoidal pulses."""

    def __init__(self, *pulses: Pulse):
        self.pulses = sorted(pulses, key=lambda p: p.t_start)

    @classmethod
    def single(cls, amplitude: float, t_hold: float, t_start: float = 0.0,
               t_rise: float = 50e-12, t_fall: float = 50e-12) -> "PulseTrain":
        return cls(Pulse(amplitude, t_start, t_hold, t_rise, t_fall))

    @classmethod
    def double(cls, amplitude: float, t_hold: float, gap: float, t_start: float = 0.0,
               t_rise: float = 50e-12, t_fall: float = 50e-12,
               t_hold2: float | None = None) -> "PulseTrain":
        p1 = Pulse(amplitude, t_start, t_hold, t_rise, t_fall)
        p2 = Pulse(amplitude, p1.t_end + gap, t_hold if t_hold2 is None else t_hold2, t_rise, t_fall)
        return cls(p1, p2)

    @property
    def t_end(self) -> float:
        return max((p.t_end for p in self.pulses), default=0.0)

    def as_array(self) -> np.ndarray:
        """(n_pulses, 5) array: t_start, t_rise, t_hold, t_fall, amplitude."""
        if not self.pulses:
            return np.zeros((0, 5))
        return np.array([p.as_row() for p in self.pulses], dtype=float)

    def __call__(self, t):
        """Voltage at time(s) t."""
        t = np.asarray(t, dtype=float)
        V = np.zeros_like(t)
        for p in self.pulses:
            t1 = p.t_start + p.t_rise
            t2 = t1 + p.t_hold
            t3 = t2 + p.t_fall
            V = np.where((t >= p.t_start) & (t < t1), p.amplitude * (t - p.t_start) / p.t_rise, V)
            V = np.where((t >= t1) & (t < t2), p.amplitude, V)
            V = np.where((t >= t2) & (t < t3), p.amplitude * (1.0 - (t - t2) / p.t_fall), V)
        return V
