"""Write (and optionally execute) the two tutorial notebooks.

    python scripts/build_notebooks.py --execute
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """\
%matplotlib inline
%config InlineBackend.figure_format = 'png'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))   # use the repo copy without installing

import numpy as np
import matplotlib.pyplot as plt
import vcma_mtj as vm
from vcma_mtj import experiments as ex, plotting as vp
vp.apply_style(9)
DATA = pathlib.Path.cwd().parent / "data"
dev = vm.deng2017()"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


def nb01():
    cells = [
        md("""# 01 — Reproducing Deng, Liang & Gupta (2017)

Macrospin model of the elliptical perpendicular MTJ free layer of
*J. Deng, G. Liang, G. Gupta, "Ultrafast and low-energy switching in voltage-controlled elliptical pMTJ", Sci. Rep. **7**, 16562 (2017)*:
a 150 × 50 × 1.7 nm³ CoFeB ellipse with a 2 nm MgO barrier, a 50 mT in-plane bias field from the bias layer, and a trapezoidal VCMA pulse of 1.6 V.

Steps
1. demagnetization factors of the elliptic-cylinder free layer (Beleggia *et al.* 2005/2006),
2. static anisotropy and the canted zero-voltage equilibria,
3. LLG dynamics for $t_{ON}$ = 0.35, 0.70 and 1.05 ns (the paper's Fig. 2),
4. how much the result depends on the initial state (switching phase diagram)."""),
        code(SETUP),
        md("""## 1. Demagnetization factors

The free layer is a thin **elliptic cylinder**, not an ellipsoid, so the Osborn ellipsoid formulas do not apply.
`vm.demag_factors` evaluates the exact Fourier-space expressions of Beleggia *et al.* (Eqs. 9–10 of the report)."""),
        code("""N = vm.demag_factors(length=150e-9, width=50e-9, thickness=1.7e-9)
print(f"Nx = {N.Nx:.6f}\\nNy = {N.Ny:.6f}\\nNz = {N.Nz:.6f}\\nsum = {sum(N):.12f}")"""),
        md("""## 2. Static anisotropy

$K_u(V) = K_{U,\\mathrm{bulk}} + [K_{I0} - \\xi V/t_{MgO}]/t_{FL}$ and
$K_{\\mathrm{eff},xz/yz}(V) = K_u(V) - K_d (N_z - N_{x/y})$ with $K_d = \\mu_0 M_s^2/2$."""),
        code("""for k, v in dev.summary().items():
    print(f"{k:26s} {v:12.5g}")"""),
        md("""With the 50 mT bias field along $x$ the two stable states are **canted** away from $\\pm z$,
with $m_x = \\mu_0 H_x / \\mu_0 H_{K,\\mathrm{eff},xz}$ (exact for a field along a principal axis):"""),
        code("""bias = (50e-3, 0, 0)
m_up, m_dn = vm.equilibrium(dev, 0, bias, +1), vm.equilibrium(dev, 0, bias, -1)
print("+z state:", m_up.round(4), " -z state:", m_dn.round(4))
print(f"cant angle = {np.degrees(np.arccos(m_up[2])):.2f} deg;  analytic m_x = {50e-3/dev.BKeff_xz(0):.4f}")"""),
        md("""## 3. Switching versus pulse ON-time (paper Fig. 2)

Following the paper's figure, the layer starts at the pole $(0,0,-1)$; $\\alpha$ = 0.01 throughout.
Eq. (8) predicts a half precession period $T/2 = \\pi(1+\\alpha^2)/(\\gamma\\mu_0H_x)$:"""),
        code("""print(f"T/2 (Eq. 8) = {dev.period_bias_only(50e-3)/2*1e9:.3f} ns")
runs = ex.run_deng(dev)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True, layout="constrained")
for ax, (t_on, r) in zip(axes, runs.items()):
    ax.axvspan(0, t_on*1e9, color="#cde2fb", lw=0)
    ax.plot(r.t_ns, r.mz, color=vp.C_MX, lw=0.9)
    ax.set(title=f"$t_{{ON}}$ = {t_on*1e9:.2f} ns  ->  {'switched' if r.switched else 'not switched'}",
           xlabel="Time (ns)", xlim=(0, 10))
axes[0].set_ylabel("$m_z$");"""),
        md("""Odd multiples of $T/2$ (0.35 and 1.05 ns) switch the layer; the even multiple (0.70 ns) returns it to the starting hemisphere — the behaviour reported by Deng *et al.*

## 4. Sensitivity to the initial state

Starting at the pole puts the magnetization 26.8° away from its real equilibrium, i.e. with extra energy.
At 1.6 V the pole is *above* the in-plane saddle, while the canted equilibrium is *below* it:"""),
        code("""for name, m in (("pole (0,0,1)", [0, 0, 1]), ("canted equilibrium", m_up)):
    print(f"E({name:18s}, 1.6 V) = {vm.energy_density(dev, m, 1.6, bias)/1e3:7.2f} kJ/m^3")
print(f"E(in-plane saddle +x,   1.6 V) = {vm.energy_density(dev, [1, 0, 0], 1.6, bias)/1e3:7.2f} kJ/m^3")"""),
        md("""The full switching phase diagram (final $m_z$ versus pulse amplitude and ON-time) makes the difference visible.
It is loaded from `data/phase_diagram_50mT.npz` (written by `scripts/make_figures.py`); set `RECOMPUTE = True` to rerun a coarse version here."""),
        code("""RECOMPUTE = False
path = DATA / "phase_diagram_50mT.npz"
if path.exists() and not RECOMPUTE:
    d = np.load(path); Vs, tons, maps = d["V"], d["t_on"], {"pole": d["mz_pole"], "eq": d["mz_eq"]}
else:
    Vs, tons = np.arange(1.0, 4.01, 0.1), np.arange(0.1e-9, 2.01e-9, 0.05e-9)
    maps = {"pole": ex.switching_map(dev, Vs, tons, bias, [0, 0, 1]),
            "eq": ex.switching_map(dev, Vs, tons, bias, m_up)}

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
for ax, key, title in zip(axes, ("pole", "eq"), ("start at the pole", "start at the canted equilibrium")):
    pc = ax.pcolormesh(tons*1e9, Vs, maps[key], cmap=vp.MZ_CMAP, norm=vp.mz_norm(), shading="nearest")
    ax.plot(0.35, 1.6, "kx", ms=8, mew=1.5)
    ax.axhline(dev.V_crit_for_bias(50e-3), color="k", lw=0.8, ls="--")
    ax.set(title=title, xlabel="$t_{ON}$ (ns)"); ax.grid(False)
axes[0].set_ylabel("$V_0$ (V)")
fig.colorbar(pc, ax=axes, label="final $m_z$ (red = switched)")
sw = maps["eq"] < 0
print(f"lowest amplitude that switches from the canted equilibrium: {Vs[np.where(sw.any(axis=1))[0][0]]:.2f} V")"""),
        md("""**Take-away.** In this macrospin model the paper's 1.6 V / 0.35 ns operating point reproduces Fig. 2 only when the free layer starts at the pole.
From the true canted equilibrium a 1.6 V pulse produces small-angle precession around the new equilibrium, and switching needs $V_0 \\gtrsim 1.84$ V —
still below the 2.61 V at which the 50 mT field removes the static barrier, because the pulse-induced jump of the equilibrium pumps energy into the precession.
Thermal fluctuations, the reference-layer stray field and STT (all omitted here) would modify these thresholds."""),
    ]
    return cells


def nb02():
    cells = [
        md("""# 02 — VCMA switching regimes

Using the corrected demagnetization factors ($N_x$ = 0.0122, $N_y$ = 0.0577, $N_z$ = 0.9301) this notebook maps how the free layer responds
to the pulse amplitude and the bias field. Each section reproduces a figure of the report.

| Regime | Voltage | Field |
|---|---|---|
| near-critical, low field | 4.8 V (just above $V_{xz}$ = 4.74 V) | $\\mu_0H_x$ = 5 mT |
| high voltage, low field | 8.5 V (above $V_{yz}$ = 7.81 V) | $\\mu_0H_x$ = 5 mT |
| high field | 0, 1.6, 3, 5 V ($V_\\mathrm{crit}$ = 2.61 V) | $\\mu_0H_x$ = 50 mT |
| VCMA-assisted field switching | 4.8 V | $\\mu_0\\mathbf{H}$ = (0.5, 0, −5) mT |

> 2 V across a 2 nm MgO barrier is already ≈1 V/nm, close to typical dielectric breakdown. The higher voltages map the macrospin regimes;
> they are not proposed operating points.

During the pulse $\\alpha$ = 0.01; from 0.1 ns after the pulse ends $\\alpha$ is raised to 0.1 so the post-pulse ring-down finishes quickly."""),
        code(SETUP),
        md("## 1. Effective anisotropy versus voltage"),
        code("""print(f"V_xz = {dev.V_xz:.3f} V   V_yz = {dev.V_yz:.3f} V   V(K_u=0) = {dev.V_Ku0:.2f} V")
V = np.linspace(-2, 10, 400)
fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
ax.plot(V, dev.BKeff_xz(V), label=r"$\\mu_0H_{K,\\mathrm{eff},xz}$")
ax.plot(V, dev.BKeff_yz(V), "--", label=r"$\\mu_0H_{K,\\mathrm{eff},yz}$")
ax.axhline(0, color=vp.MUTED, lw=0.6)
for v in (dev.V_xz, dev.V_yz): ax.axvline(v, color=vp.INK2, lw=0.6, ls=":")
ax.set(xlabel="Voltage (V)", ylabel=r"$\\mu_0H_{K,\\mathrm{eff}}$ (T)"); ax.legend();"""),
        md("""Threshold in-plane field that flattens the $x$–$z$ barrier, $\\mu_0 H_{x,\\mathrm{th}}(V) = 2K_{\\mathrm{eff},xz}(V)/M_s$ (Eq. 13),
and the easy-axis switching field of Eq. (14):"""),
        code("""for V0 in (0, 2.5, 4.53, 4.8):
    print(f"V = {V0:4.2f} V:  mu0 H_x,th = {1e3*dev.B_threshold_x(V0):6.1f} mT   mu0 H_sw,z = {1e3*dev.B_switch_z(V0):6.1f} mT")"""),
        md("## 2. Energy landscapes at zero field"),
        code("""th = np.linspace(0, np.pi, 361)
m = vm.unit_vector(th, 0*th)
fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
for V0 in ex.LANDSCAPE_VOLTAGES:
    ax.plot(np.degrees(th), vm.energy_density(dev, m, V0)/1e3, label=f"{V0:g} V")
ax.set(xlabel=r"$\\theta$ (deg), $\\phi$ = 0", ylabel=r"E (kJ/m$^3$)", xlim=(0, 180)); ax.legend(ncol=4);

fig = plt.figure(figsize=(11, 2.8))
for k, V0 in enumerate(ex.LANDSCAPE_VOLTAGES):
    ax = fig.add_subplot(1, 4, k + 1, projection="3d"); vp.sphere(ax, dev, V0, n=100); ax.set_title(f"{V0:g} V")"""),
        md("""## 3. Near-critical voltage, low field (4.8 V, 5 mT)

A long (9 ns) pulse reveals the precession period about the in-plane field."""),
        code("""NC = ex.NearCritical()
r = ex.run_pulse(dev, NC.V0, NC.t_hold_long, (NC.Bx, 0, 0))
T = ex.measured_period(r, ex.T_START, ex.T_START + NC.t_hold_long)
print(f"mean precession period during the pulse: {T*1e9:.2f} ns")
vp.plot_dynamics(r, (9, 4.5), title="4.8 V, 5 mT, 9 ns pulse");"""),
        code("""for th in (NC.t_hold_half, NC.t_hold_quarter):
    r = ex.run_pulse(dev, NC.V0, th, (NC.Bx, 0, 0))
    vp.plot_dynamics(r, (9, 3.6), title=f"t_hold = {th*1e9:g} ns  ->  final m_z = {r.final[2]:+.3f}")
r = ex.run_pulse(dev, NC.V0, NC.t_hold_half, (NC.Bx, 0, 0), double_gap=NC.gap)
vp.plot_dynamics(r, (9, 3.6), title="two 1 ns pulses: +z -> -z -> +z");"""),
        md("""Switching windows: the layer switches when the pulse ends while $m$ is in the opposite hemisphere, i.e. for
$(n+\\tfrac14)T < t < (n+\\tfrac34)T$:"""),
        md("""Shaded: where the simulation actually switched. Dashed: the rule, using the period fitted to the windows.
(The mean period on the single 9 ns pulse above is a little shorter, so it drifts off the windows after a few periods.)"""),
        code("""d = np.load(DATA / "nearcrit_window_scan.npz")
wins = ex.windows_from_scan(d["t_hold"], d["mz_final"])
Tw = ex.window_period(wins)
print("windows (ns):", [(round(float(a)*1e9, 2), round(float(b)*1e9, 2)) for a, b in wins])
print(f"period fitted to the windows: {Tw*1e9:.2f} ns  (single-pulse mean: {T*1e9:.2f} ns)")
fig, ax = plt.subplots(figsize=(9, 2.6), layout="constrained")
for a, b in wins: ax.axvspan(a*1e9, b*1e9, color="#cde2fb", lw=0)
for n in range(len(wins)):
    for f in (.25, .75): ax.axvline((n+f)*Tw*1e9, color="0.5", ls="--", lw=0.8)
ax.plot(d["t_hold"]*1e9, d["mz_final"], ".", ms=3)
ax.set(xlabel="$t_{hold}$ (ns)", ylabel="final $m_z$", xlim=(0, 6));"""),
        md("""## 4. High voltage, low field (8.5 V, 5 mT)

Above $V_{yz}$ both out-of-plane barriers are inverted: $\\pm z$ become maxima, $\\pm x$ minima and $\\pm y$ saddles.
$m$ spirals away from $+z$ around the $z$ axis, so switching is delayed by >13 ns."""),
        code("""HV = ex.HighVoltage()
for th, case in ((HV.t_hold_case1, "i"), (HV.t_hold_case2, "ii")):
    r = ex.run_pulse(dev, HV.V0, th, (HV.Bx, 0, 0))
    print(f"case ({case}): t_hold = {th*1e9:g} ns, min m_z during pulse = {r.mz[r.t <= ex.T_START + th].min():+.3f}, final m_z = {r.final[2]:+.3f}")
    vp.plot_dynamics(r, (9, 3.6), title=f"8.5 V, case ({case})")
fig = plt.figure(figsize=(5, 5)); vp.trajectory_3d(fig.add_subplot(projection="3d"), r)"""),
        code("""d = np.load(DATA / "highV_window_scan.npz")
fig, ax = plt.subplots(figsize=(9, 2.4), layout="constrained")
ax.plot(d["t_hold"]*1e9, d["mz_final"], ".", ms=3)
ax.set(xlabel="$t_{hold}$ (ns)", ylabel="final $m_z$", title="switching windows are only 0.1-0.5 ns wide");"""),
        md("""## 5. High field (50 mT)

The field alone removes the barrier once $\\mu_0H_{K,\\mathrm{eff},xz}(V) < 50$ mT, i.e. above 2.61 V; above it $m$ orbits the $x$ axis."""),
        code("""HF = ex.HighField(); bias = (HF.Bx, 0, 0)
m0 = vm.equilibrium(dev, 0, bias, +1)
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True, layout="constrained")
for ax, V0 in zip(axes, HF.voltages):
    r = ex.run_pulse(dev, V0, HF.t_hold, bias, m0=m0, post=4e-9)
    vp.landscape_2d(ax, dev, V0, bias, res=r); ax.set_title(f"{V0:g} V: final m_z = {r.final[2]:+.2f}")
    ax.set_xlabel(r"$\\theta$ (deg)")
axes[0].set_ylabel(r"$\\phi$ (deg)");"""),
        md("""## 6. VCMA-assisted field-driven switching

Without a voltage, a $-z$ field must exceed $\\mu_0H_{sw}(0)$ = 111 mT (Eq. 14). At 4.8 V this drops to about −1 mT, so a −5 mT field
(plus 0.5 mT along $x$ to break the symmetry) switches the layer, and the final state is set by the field, not by the pulse width."""),
        code("""FA = ex.FieldAssisted()
print(f"mu0 H_sw(0) = {1e3*dev.B_switch_z(0):.1f} mT,  mu0 H_sw({FA.V0} V) = {1e3*dev.B_switch_z(FA.V0):.1f} mT")
r = ex.run_pulse(dev, FA.V0, FA.t_hold, FA.bias, double_gap=FA.gap, post=5e-9)
vp.plot_dynamics(r, (9, 3.6), title="4.8 V, (0.5, 0, -5) mT, two 15 ns pulses");
d = np.load(DATA / "field_assisted_scan.npz")
th, wb = d["t_hold"], d["mz_with_bz"]
first = next(t for t in th if np.all(wb[th >= t] < 0))
print(f"with -5 mT the outcome is -z for every pulse longer than {first*1e9:.1f} ns;"
      f" without it the outcome keeps alternating ({np.sum(np.diff(np.sign(d['mz_no_bz'])) != 0)} flips over 0-20 ns)")"""),
    ]
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    NB.mkdir(exist_ok=True)
    for name, cells in (("01_deng2017_reproduction.ipynb", nb01()), ("02_vcma_switching_regimes.ipynb", nb02())):
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB / name
        if args.execute:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=900, kernel_name="python3",
                           resources={"metadata": {"path": str(NB)}}).execute()
        nbf.write(nb, path)
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
