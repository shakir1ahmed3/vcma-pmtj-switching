"""Regenerate every figure in ``figures/`` (and therefore in the report).

    python scripts/make_figures.py            # everything (~3-4 min with numba)
    python scripts/make_figures.py --only fig10 fig11
    python scripts/make_figures.py --quick    # coarser switching maps

Figure sizes (inches) match the picture frames of the Word report exactly,
so the PNGs can be swapped into the document without distortion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vcma_mtj import deng2017, energy_density, energy_map, equilibrium, unit_vector  # noqa: E402
from vcma_mtj import experiments as ex  # noqa: E402
from vcma_mtj import plotting as vp  # noqa: E402

FIG = ROOT / "figures"
DATA = ROOT / "data"
DEV = deng2017()
NEAR, HIGHV, HIGHF, FA = ex.NearCritical(), ex.HighVoltage(), ex.HighField(), ex.FieldAssisted()
RESULTS: dict = {}  # numbers quoted in the report / README, written to data/results.json
RECOMPUTE = False  # set by --recompute; otherwise parameter scans are loaded from data/*.npz


def cached(name: str, compute):
    """Load a parameter scan from data/<name>.npz, or compute and store it."""
    path = DATA / f"{name}.npz"
    if path.exists() and not RECOMPUTE:
        return dict(np.load(path))
    out = compute()
    np.savez(path, **out)
    return out


def ns(x):
    return f"{x * 1e9:g} ns"


# ----------------------------------------------------------------------
# A. Deng et al. reproduction
# ----------------------------------------------------------------------
def fig03():
    case = ex.DengCase()
    runs = ex.run_deng(DEV, case)
    fig, axes = plt.subplots(1, 3, figsize=(6.50, 1.94), sharey=True, layout="constrained")
    for ax, (t_on, r) in zip(axes, runs.items()):
        ax.axvspan(0, t_on * 1e9, color="#cde2fb", lw=0, zorder=0)
        ax.plot(r.t_ns, r.mz, color=vp.C_MX, lw=0.9)
        ax.set_title(rf"$t_{{ON}}$ = {t_on * 1e9:.2f} ns")
        ax.set_xlim(0, 10)
        ax.set_xlabel("Time (ns)")
        RESULTS.setdefault("deng", {})[f"{t_on * 1e9:.2f}"] = {"final_mz": float(r.final[2]), "switched": r.switched}
    axes[0].set_ylabel(r"$m_z$")
    axes[0].set_ylim(-1.1, 1.1)
    vp.save(fig, FIG / "fig03_deng_reproduction")
    RESULTS["deng"]["T_half_eq8_ns"] = DEV.period_bias_only(case.Bx) / 2 * 1e9


# ----------------------------------------------------------------------
# B. Effect of voltage on anisotropy
# ----------------------------------------------------------------------
def fig04():
    V = np.linspace(-5, 75, 801)
    fig, ax = plt.subplots(figsize=(4.51, 2.74), layout="constrained")
    ax.plot(V, DEV.BK(V), color=vp.C_MX, label=r"$\mu_0H_{K,u}$ (uniaxial)")
    ax.plot(V, DEV.BKeff_xz(V), color=vp.C_MY, label=r"$\mu_0H_{K,\mathrm{eff},xz}$")
    ax.plot(V, DEV.BKeff_yz(V), color=vp.C_MZ, ls="--", label=r"$\mu_0H_{K,\mathrm{eff},yz}$")
    ax.axhline(0, color=vp.MUTED, lw=0.6)
    for v, txt, xy in ((DEV.V_xz, rf"$V_{{xz}}$ = {DEV.V_xz:.2f} V", (-3.5, -0.75)),
                       (DEV.V_yz, rf"$V_{{yz}}$ = {DEV.V_yz:.2f} V", (12, 0.42)),
                       (DEV.V_Ku0, rf"$V_{{K_u=0}}$ = {DEV.V_Ku0:.1f} V", (50, -0.55))):
        ax.plot(v, 0, "o", ms=3.5, color=vp.INK, zorder=5)
        ax.annotate(txt, (v, 0), xy, fontsize=6.5, color=vp.INK,
                    arrowprops=dict(arrowstyle="-", color=vp.INK2, lw=0.6))
    ax.set_xlim(-5, 75)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(r"$\mu_0 H_K$ (T)")
    ax.legend(loc="upper right")
    vp.save(fig, FIG / "fig04_anisotropy_fields_vs_voltage")


def fig05():
    V = np.linspace(-2, 10, 601)
    fig, ax = plt.subplots(figsize=(5.07, 3.08), layout="constrained")
    v1, v2 = DEV.V_xz, DEV.V_yz
    for lo, hi, lab in ((-2, v1, "I"), (v1, v2, "II"), (v2, 10, "III")):
        ax.axvspan(lo, hi, color="#f0efec" if lab != "II" else "#e4e3df", lw=0, zorder=0)
        ax.text((lo + hi) / 2, 0.205, lab, ha="center", va="center", color=vp.INK2, fontsize=8)
    ax.plot(V, DEV.BKeff_xz(V), color=vp.C_MX, label=r"$\mu_0H_{K,\mathrm{eff},xz}$")
    ax.plot(V, DEV.BKeff_yz(V), color=vp.C_MY, ls="--", label=r"$\mu_0H_{K,\mathrm{eff},yz}$")
    ax.axhline(0, color=vp.MUTED, lw=0.6)
    for v in (v1, v2):
        ax.axvline(v, color=vp.INK2, lw=0.6, ls=":")
    ax.annotate(f"{v1:.2f} V", (v1, 0), (v1 - 1.9, -0.075), fontsize=7,
                arrowprops=dict(arrowstyle="-", color=vp.INK2, lw=0.6))
    ax.annotate(f"{v2:.2f} V", (v2, 0), (v2 + 0.35, 0.05), fontsize=7,
                arrowprops=dict(arrowstyle="-", color=vp.INK2, lw=0.6))
    ax.set_xlim(-2, 10)
    ax.set_ylim(-0.12, 0.225)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(r"$\mu_0 H_{K,\mathrm{eff}}$ (T)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.92))
    vp.save(fig, FIG / "fig05_effective_anisotropy_xz_yz")
    RESULTS["critical"] = {
        "Nx": DEV.Nx, "Ny": DEV.Ny, "Nz": DEV.Nz,
        "V_c": DEV.V_c, "V_xz": v1, "V_yz": v2, "V_Ku0": DEV.V_Ku0,
        "BKeff0_mT": 1e3 * float(DEV.BKeff(0)), "BKeff_xz0_mT": 1e3 * float(DEV.BKeff_xz(0)),
        "BKeff_yz0_mT": 1e3 * float(DEV.BKeff_yz(0)),
        "Vcrit_5mT": DEV.V_crit_for_bias(5e-3), "Vcrit_50mT": DEV.V_crit_for_bias(50e-3),
        "Bsw_z0_mT": 1e3 * float(DEV.B_switch_z(0)),
    }


def fig06():
    th = np.linspace(0, np.pi, 721)
    m = unit_vector(th, np.zeros_like(th))
    fig, ax = plt.subplots(figsize=(5.32, 2.97), layout="constrained")
    for V, c in zip(ex.LANDSCAPE_VOLTAGES, vp.SERIES):
        ax.plot(np.degrees(th), energy_density(DEV, m, V) / 1e3, color=c, label=f"V = {V:g} V")
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.set_xlabel(r"Polar angle $\theta$ (deg), $\phi = 0$ (x–z plane)")
    ax.set_ylabel(r"Energy density (kJ/m$^3$)")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12))
    vp.save(fig, FIG / "fig06_energy_profile_1d")


def _landscape_row(voltages, bias, figsize, name, runs=None):
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(1, len(voltages) + 1, width_ratios=[1] * len(voltages) + [0.05])
    E_all = []
    th = np.linspace(0, np.pi, 91)
    ph = np.linspace(-np.pi / 2, np.pi / 2, 91)
    TH, PH = np.meshgrid(th, ph)
    for V in voltages:
        E_all.append(energy_map(DEV, TH, PH, V, bias) / 1e3)
    vlim = (min(e.min() for e in E_all), max(e.max() for e in E_all))
    levels = np.linspace(*vlim, 22)
    for k, V in enumerate(voltages):
        ax = fig.add_subplot(gs[0, k])
        cf = vp.landscape_2d(ax, DEV, V, bias, res=None if runs is None else runs[V], levels=levels, vlim=vlim)
        ax.set_title(f"V = {V:g} V", fontsize=7.5)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel(r"$\phi$ (deg)", fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r"$\theta$ (deg)", fontsize=7, labelpad=1)
    cax = fig.add_subplot(gs[0, -1])
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label(r"E (kJ/m$^3$)", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.4)
    vp.save(fig, FIG / name)


def fig07():
    _landscape_row(ex.LANDSCAPE_VOLTAGES, (0, 0, 0), (6.50, 1.80), "fig07_energy_landscape_2d")


def _sphere_row(voltages, bias, figsize, name, runs=None, title=True):
    fig = plt.figure(figsize=figsize)
    n = len(voltages)
    for k, V in enumerate(voltages):
        ax = fig.add_axes([k / n, 0.0, 1 / n, 0.86 if title else 1.0], projection="3d")
        vp.sphere(ax, DEV, V, bias, res=None if runs is None else runs[V])
        if title:
            fig.text((k + 0.5) / n, 0.93, f"V = {V:g} V", ha="center", va="center", fontsize=8)
    vp.save(fig, FIG / name, pdf=False)


def fig08():
    _sphere_row(ex.LANDSCAPE_VOLTAGES, (0, 0, 0), (5.74, 1.49), "fig08_energy_landscape_3d")


def fig09():
    V = np.linspace(-5, 6, 551)
    fig, ax = plt.subplots(figsize=(3.75, 2.33), layout="constrained")
    ax.plot(V, 1e3 * DEV.B_threshold_x(V), color=vp.C_MX)
    ax.plot(0, 1e3 * DEV.B_threshold_x(0), "o", ms=3.5, color=vp.INK)
    ax.annotate(f"{1e3 * DEV.B_threshold_x(0):.0f} mT at 0 V", (0, 1e3 * DEV.B_threshold_x(0)),
                (0.5, 1e3 * DEV.B_threshold_x(0) + 25), fontsize=7,
                arrowprops=dict(arrowstyle="-", color=vp.INK2, lw=0.6))
    ax.plot(DEV.V_xz, 0, "o", ms=3.5, color=vp.INK)
    ax.annotate(f"0 at {DEV.V_xz:.2f} V", (DEV.V_xz, 0), (DEV.V_xz - 3.6, 18), fontsize=7,
                arrowprops=dict(arrowstyle="-", color=vp.INK2, lw=0.6))
    ax.set_xlim(-5, 6)
    ax.set_ylim(0, None)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(r"$\mu_0 H_{x,\mathrm{th}}$ (mT)")
    vp.save(fig, FIG / "fig09_threshold_inplane_field")


# ----------------------------------------------------------------------
# D. Near-critical voltage, low field
# ----------------------------------------------------------------------
def _near(t_hold, figsize, name, gap=None):
    r = ex.run_pulse(DEV, NEAR.V0, t_hold, (NEAR.Bx, 0, 0), double_gap=gap)
    kind = "two pulses" if gap else "pulse"
    fig, (ax, _) = vp.plot_dynamics(
        r, figsize, title=rf"$V_0$ = {NEAR.V0:g} V, $\mu_0H_x$ = {NEAR.Bx * 1e3:g} mT, "
                          rf"$t_{{hold}}$ = {t_hold * 1e9:g} ns ({kind})")
    vp.save(fig, FIG / name)
    return r


def fig10():
    r = _near(NEAR.t_hold_long, (5.86, 3.64), "fig10_nearcrit_long_pulse")
    T = ex.measured_period(r, ex.T_START, ex.T_START + NEAR.t_hold_long)
    zc = np.where(np.diff(np.sign(r.mz)) != 0)[0]
    RESULTS["near_critical"] = {"V0": NEAR.V0, "Bx_mT": NEAR.Bx * 1e3, "period_ns": T * 1e9,
                                "first_zero_crossing_after_pulse_start_ns": (r.t[zc[0]] - ex.T_START) * 1e9}


def fig11():
    _near(NEAR.t_hold_half, (4.91, 3.18), "fig11_nearcrit_half_period")


def fig12():
    _near(NEAR.t_hold_quarter, (5.42, 3.42), "fig12_nearcrit_quarter_period")


def fig13():
    r = _near(NEAR.t_hold_half, (5.58, 3.17), "fig13_nearcrit_double_pulse", gap=NEAR.gap)
    RESULTS.setdefault("near_critical", {})["double_pulse_final_mz"] = float(r.final[2])


def figS_window(quick=False):
    """Final m_z vs flat-top duration at the near-critical point (README/notebook)."""
    step = 0.05e-9 if quick else 0.02e-9

    def compute():
        t_holds = np.arange(0.02e-9, 6.0e-9 + 1e-15, step)
        return {"t_hold": t_holds, "mz_final": ex.switching_vs_width(DEV, NEAR.V0, t_holds, (NEAR.Bx, 0, 0))}

    d = cached("nearcrit_window_scan", compute)
    t_holds, mz = d["t_hold"], d["mz_final"]
    wins = ex.windows_from_scan(t_holds, mz)
    # Shade the simulated windows themselves; overlay the (n+1/4)T..(n+3/4)T rule with the
    # period fitted to those windows. (The mean period on one 9 ns pulse, ~1.85 ns, drifts
    # off the windows by ~0.15 ns after three periods, so it is not used here.)
    T = ex.window_period(wins) * 1e9
    fig, ax = plt.subplots(figsize=(6.5, 2.3), layout="constrained")
    for i, (a, b) in enumerate(wins):
        ax.axvspan(a * 1e9, b * 1e9, color="#cde2fb", lw=0, zorder=0,
                   label="simulated switching window" if i == 0 else None)
    for n in range(len(wins)):
        for k, f in enumerate((0.25, 0.75)):
            ax.axvline((n + f) * T, color=vp.MUTED, lw=0.8, ls="--", zorder=1,
                       label=rf"$(n+\frac{{1}}{{4}})T$, $(n+\frac{{3}}{{4}})T$, $T$ = {T:.2f} ns"
                       if n == 0 and k == 0 else None)
    ax.plot(t_holds * 1e9, mz, ".", ms=2.5, color=vp.C_MX, label="simulated final $m_z$", zorder=2)
    ax.axhline(0, color=vp.MUTED, lw=0.6)
    ax.set_xlim(0, 6)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r"Pulse flat-top duration $t_{hold}$ (ns)")
    ax.set_ylabel(r"final $m_z$")
    ax.set_title(rf"Switching windows at $V_0$ = {NEAR.V0:g} V, $\mu_0H_x$ = 5 mT (start +z)", loc="left")
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.12), frameon=False)
    RESULTS.setdefault("near_critical", {})["window_period_ns"] = T
    vp.save(fig, FIG / "figS1_nearcrit_switching_windows")
    RESULTS.setdefault("near_critical", {})["windows_ns"] = [(a * 1e9, b * 1e9) for a, b in wins]


# ----------------------------------------------------------------------
# E. High voltage, low field
# ----------------------------------------------------------------------
def _highv(t_hold, figsize, name, case):
    r = ex.run_pulse(DEV, HIGHV.V0, t_hold, (HIGHV.Bx, 0, 0), store_every=10)
    fig, _ = vp.plot_dynamics(r, figsize, title=rf"$V_0$ = {HIGHV.V0:g} V, $\mu_0H_x$ = 5 mT, "
                                                rf"$t_{{hold}}$ = {t_hold * 1e9:g} ns — case ({case})")
    vp.save(fig, FIG / name)
    return r


def fig14_15():
    r = _highv(HIGHV.t_hold_case1, (5.21, 3.06), "fig14_highV_case1", "i")
    idx = np.where(r.mz < 0)[0]
    RESULTS["high_voltage"] = {"V0": HIGHV.V0, "first_mz_zero_crossing_ns": float(r.t[idx[0]] * 1e9),
                               "case1_t_hold_ns": HIGHV.t_hold_case1 * 1e9,
                               "case1_final_mz": float(r.final[2])}
    # left: energy sphere at 8.5 V (no title), right: 3-D trajectory
    fig = plt.figure(figsize=(2.59, 2.26))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    vp.sphere(ax, DEV, HIGHV.V0, (HIGHV.Bx, 0, 0))
    vp.save(fig, FIG / "fig15a_sphere_highV", pdf=False)
    fig = plt.figure(figsize=(3.13, 3.13))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d")
    vp.trajectory_3d(ax, r)
    vp.save(fig, FIG / "fig15b_traj3d_highV_case1", pdf=False)


def fig16_17():
    r = _highv(HIGHV.t_hold_case2, (6.42, 4.13), "fig16_highV_case2", "ii")
    RESULTS["high_voltage"].update({"case2_t_hold_ns": HIGHV.t_hold_case2 * 1e9,
                                    "case2_final_mz": float(r.final[2]),
                                    "case2_min_mz_during_pulse": float(r.mz[r.t <= ex.T_START + HIGHV.t_hold_case2].min())})
    for V, size, name in ((HIGHV.V0, (1.61, 2.30), "fig17a_sphere_highV"), (0.0, (1.90, 2.41), "fig17b_sphere_0V")):
        fig = plt.figure(figsize=size)
        ax = fig.add_axes([0, 0, 1, 0.88], projection="3d")
        vp.sphere(ax, DEV, V, (HIGHV.Bx, 0, 0))
        fig.text(0.5, 0.93, f"V = {V:g} V", ha="center", fontsize=8)
        vp.save(fig, FIG / name, pdf=False)
    fig = plt.figure(figsize=(3.97, 3.65))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d")
    vp.trajectory_3d(ax, r)
    vp.save(fig, FIG / "fig17c_traj3d_highV_case2", pdf=False)


def figS_highv_scan(quick=False):
    step = 0.2e-9 if quick else 0.05e-9

    def compute():
        t_holds = np.arange(5e-9, 30e-9 + 1e-15, step)
        return {"t_hold": t_holds, "mz_final": ex.switching_vs_width(DEV, HIGHV.V0, t_holds, (HIGHV.Bx, 0, 0))}

    d = cached("highV_window_scan", compute)
    t_holds, mz = d["t_hold"], d["mz_final"]
    t_fine = np.arange(t_holds.min(), t_holds.max() + 1e-15, 2e-12)
    mz_off = ex.mz_at_pulse_off(DEV, HIGHV.V0, t_fine, (HIGHV.Bx, 0, 0))
    fig, ax = plt.subplots(figsize=(6.5, 2.4), layout="constrained")
    ax.plot(t_fine * 1e9, mz_off, color=vp.C_MX, lw=0.6, alpha=0.3, label=r"$m_z$ when the pulse switches off")
    ax.plot(t_holds * 1e9, mz, ".", ms=2.5, color=vp.C_MX, label=r"final $m_z$")
    ax.axhline(0, color=vp.MUTED, lw=0.6)
    for th, c in ((HIGHV.t_hold_case2, "ii"), (HIGHV.t_hold_case1, "i")):
        ax.axvline(th * 1e9, color=vp.INK2, lw=0.6, ls=":")
        ax.text(th * 1e9 + 0.15, 0.55, f"case ({c})", fontsize=7, color=vp.INK2)
    ax.set_xlim(5, 30)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r"Pulse flat-top duration $t_{hold}$ (ns)")
    ax.set_ylabel(r"$m_z$")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.12), frameon=False)
    ax.set_title(rf"High-voltage regime ($V_0$ = {HIGHV.V0:g} V, 5 mT): switching windows only 0.1–0.5 ns wide",
                 loc="left")
    vp.save(fig, FIG / "figS2_highV_switching_scan")
    RESULTS.setdefault("high_voltage", {})["first_window_ns"] = (
        ex.windows_from_scan(t_holds, mz)[0] if (mz < 0).any() else None)
    RESULTS["high_voltage"]["first_window_ns"] = [x * 1e9 for x in RESULTS["high_voltage"]["first_window_ns"]]


# ----------------------------------------------------------------------
# F. High field
# ----------------------------------------------------------------------
def fig18_19():
    bias = (HIGHF.Bx, 0, 0)
    m0 = equilibrium(DEV, 0.0, bias, +1)
    runs = {V: ex.run_pulse(DEV, V, HIGHF.t_hold, bias, m0=m0, post=4e-9, store_every=10) for V in HIGHF.voltages}
    _landscape_row(HIGHF.voltages, bias, (6.50, 1.47), "fig18_highfield_landscape_2d", runs=runs)
    _sphere_row(HIGHF.voltages, bias, (5.87, 1.52), "fig19_highfield_landscape_3d", runs=runs)
    RESULTS["high_field"] = {"Bx_mT": HIGHF.Bx * 1e3, "V_crit": DEV.V_crit_for_bias(HIGHF.Bx),
                             "theta_eq0_deg": float(np.degrees(np.arccos(m0[2]))),
                             "final_mz": {f"{V:g}": float(r.final[2]) for V, r in runs.items()}}


# ----------------------------------------------------------------------
# G. VCMA-assisted field-driven switching
# ----------------------------------------------------------------------
def fig20():
    r = ex.run_pulse(DEV, FA.V0, FA.t_hold, FA.bias, double_gap=FA.gap, post=5e-9, store_every=10)
    fig, _ = vp.plot_dynamics(r, (5.63, 3.18),
                              title=rf"$V_0$ = {FA.V0:g} V, $\mu_0\mathbf{{H}}$ = (0.5, 0, −5) mT, two {FA.t_hold * 1e9:g} ns pulses")
    vp.save(fig, FIG / "fig20_field_assisted_double_pulse")
    p1_end = r.pulses.pulses[0].t_end
    RESULTS["field_assisted"] = {"V0": FA.V0, "bias_mT": [b * 1e3 for b in FA.bias],
                                 "Bsw_z0_mT": 1e3 * float(DEV.B_switch_z(0)),
                                 "Bsw_z_V0_mT": 1e3 * float(DEV.B_switch_z(FA.V0)),
                                 "mz_after_pulse1": float(r.mz[np.searchsorted(r.t, p1_end + 4e-9)]),
                                 "final_mz": float(r.final[2])}


def figS_field_assisted(quick=False):
    step = 0.25e-9 if quick else 0.1e-9

    def compute():
        t_holds = np.arange(0.1e-9, 20e-9 + 1e-15, step)
        return {"t_hold": t_holds,
                "mz_with_bz": ex.switching_vs_width(DEV, FA.V0, t_holds, FA.bias),
                "mz_no_bz": ex.switching_vs_width(DEV, FA.V0, t_holds, (FA.bias[0], 0, 0))}

    d = cached("field_assisted_scan", compute)
    t_holds, with_bz, no_bz = d["t_hold"], d["mz_with_bz"], d["mz_no_bz"]
    fig, ax = plt.subplots(figsize=(6.5, 1.75), layout="constrained")
    grid = np.vstack([no_bz, with_bz])
    ax.pcolormesh(t_holds * 1e9, [0, 1], grid, cmap=vp.MZ_CMAP, norm=vp.mz_norm(1.0), shading="nearest")
    ax.set_yticks([0, 1], [r"$\mu_0H_z$ = 0" + "\n(pure VCMA)", r"$\mu_0H_z$ = −5 mT" + "\n(VCMA-assisted)"])
    for y in (0.5,):
        ax.axhline(y, color="white", lw=2)
    ax.set_xlim(t_holds[0] * 1e9, 20)
    ax.set_xlabel(r"Pulse flat-top duration $t_{hold}$ (ns)")
    ax.set_title(rf"Final state after one {FA.V0:g} V pulse (red = switched to −z, blue = stayed +z)", loc="left")
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    vp.save(fig, FIG / "figS3_field_assisted_determinism")
    always = None
    for th in t_holds:
        if np.all(with_bz[t_holds >= th] < 0):
            always = float(th)
            break
    RESULTS.setdefault("field_assisted", {})["deterministic_from_ns"] = None if always is None else always * 1e9


# ----------------------------------------------------------------------
# Headline: switching phase diagram under Deng conditions
# ----------------------------------------------------------------------
def figS_phase(quick=False):
    Vs = np.arange(1.0, 4.0 + 1e-9, 0.1 if quick else 0.04)
    tons = np.arange(0.10e-9, 2.0e-9 + 1e-15, 0.05e-9 if quick else 0.02e-9)
    bias = (50e-3, 0, 0)
    m_pole = np.array([0.0, 0.0, 1.0])
    m_eq = equilibrium(DEV, 0.0, bias, +1)

    def compute():
        return {"V": Vs, "t_on": tons,
                "mz_pole": ex.switching_map(DEV, Vs, tons, bias, m_pole),
                "mz_eq": ex.switching_map(DEV, Vs, tons, bias, m_eq)}

    d = cached("phase_diagram_50mT", compute)
    Vs, tons = d["V"], d["t_on"]
    maps = {"pole": d["mz_pole"], "eq": d["mz_eq"]}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True, layout="constrained")
    titles = {"pole": r"start at the pole $(0,0,1)$ — as in Deng et al.",
              "eq": rf"start at the canted equilibrium ($\theta$ = {np.degrees(np.arccos(m_eq[2])):.1f}°)"}
    for ax, key in zip(axes, ("pole", "eq")):
        pc = ax.pcolormesh(tons * 1e9, Vs, maps[key], cmap=vp.MZ_CMAP, norm=vp.mz_norm(1.0),
                           shading="nearest", rasterized=True)
        ax.plot([0.35], [1.6], marker="x", ms=6, mew=1.2, color=vp.INK)
        ax.axhline(DEV.V_crit_for_bias(0.05), color=vp.INK, lw=0.7, ls="--")
        ax.set_title(titles[key], fontsize=7.5, loc="left")
        ax.set_xlabel(r"$t_{ON}$ (ns)")
        ax.grid(False)
    axes[0].set_ylabel("Pulse amplitude $V_0$ (V)")
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)
    for ax in axes:
        ax.text(1.98, DEV.V_crit_for_bias(0.05) + 0.05, rf"static barrier gone ({DEV.V_crit_for_bias(0.05):.2f} V)",
                ha="right", va="bottom", fontsize=6.5, bbox=box)
        ax.text(0.40, 1.6, "Deng et al.\n1.6 V, 0.35 ns", ha="left", va="center", fontsize=6.5, bbox=box)
    cb = fig.colorbar(pc, ax=axes, shrink=0.9, pad=0.01)
    cb.set_label(r"final $m_z$ (red = switched)")
    cb.outline.set_linewidth(0.4)
    vp.save(fig, FIG / "figS4_switching_phase_diagram_50mT")
    eq_sw = maps["eq"] < 0
    rows = np.where(eq_sw.any(axis=1))[0]
    RESULTS["phase_diagram"] = {"min_V_switch_from_eq": float(Vs[rows[0]]) if rows.size else None,
                                "V_step": float(Vs[1] - Vs[0])}


ALL = {
    "fig03": fig03, "fig04": fig04, "fig05": fig05, "fig06": fig06, "fig07": fig07, "fig08": fig08,
    "fig09": fig09, "fig10": fig10, "fig11": fig11, "fig12": fig12, "fig13": fig13,
    "fig14": fig14_15, "fig16": fig16_17, "fig18": fig18_19, "fig20": fig20,
    "figS1": figS_window, "figS2": figS_highv_scan, "figS3": figS_field_assisted, "figS4": figS_phase,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="subset of figure keys, e.g. fig10 figS4")
    ap.add_argument("--quick", action="store_true", help="coarser parameter scans (with --recompute)")
    ap.add_argument("--recompute", action="store_true", help="re-run cached parameter scans in data/")
    args = ap.parse_args()
    global RECOMPUTE
    RECOMPUTE = args.recompute
    vp.apply_style()
    FIG.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    res_path = DATA / "results.json"
    if res_path.exists():
        RESULTS.update(json.loads(res_path.read_text()))
    keys = args.only or list(ALL)
    for k in keys:
        t0 = time.time()
        fn = ALL[k]
        if k.startswith("figS"):
            fn(quick=args.quick)
        else:
            fn()
        print(f"{k:6s} done in {time.time() - t0:5.1f} s")
    res_path.write_text(json.dumps(RESULTS, indent=2, default=float))
    print(f"numbers written to {res_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
