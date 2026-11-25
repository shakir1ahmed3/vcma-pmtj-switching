# Changelog

## 1.0.0 (2026-11-24)

- Simulation code refactored from the course notebooks into the `vcma_mtj` package, with a numba-compiled RK4
  integrator, unit tests and a single script (`scripts/make_figures.py`) that regenerates every figure.
- Demagnetization factors now taken from the exact elliptic-cylinder expressions of Beleggia et al. throughout
  (N<sub>x</sub> = 0.012241, N<sub>y</sub> = 0.057682, N<sub>z</sub> = 0.930077). The course-report version used
  N<sub>x</sub> = 0.011, N<sub>y</sub> = 0.034, N<sub>z</sub> = 0.955 in the regime studies. With the corrected values the
  in-plane barriers vanish at 4.74 V and 7.81 V (previously 2.98 V and 4.55 V), so the near-critical and high-voltage
  cases now run at 4.8 V and 8.5 V. All figures and the report were regenerated.
- Gilbert damping during pulses set to α = 0.01 everywhere (consistent with the report table), and γ to the CODATA
  electron value.
- Eq. (13) of the report corrected to H<sub>x,th</sub> = 2K<sub>eff,xz</sub>/(μ<sub>0</sub>M<sub>s</sub>).
- New analyses: switching-window scans, the 50 mT switching phase diagram (pole versus canted-equilibrium start), and
  a determinism scan for VCMA-assisted field switching.
