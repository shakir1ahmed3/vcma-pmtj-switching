"""Plot helpers with one consistent, print-friendly style."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .device import Device
from .energy import energy_density, energy_map, unit_vector
from .llg import SimResult

# categorical slots (fixed order) and neutral inks
C_MX = "#2a78d6"  # blue
C_MY = "#eb6834"  # orange
C_MZ = "#1baf7a"  # aqua
C_4 = "#eda100"  # yellow
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
TRAJ = "#eb6834"

COMPONENT_COLORS = {"mx": C_MX, "my": C_MY, "mz": C_MZ}
SERIES = [C_MX, C_MY, C_MZ, C_4]

# single-hue sequential ramp (light = high energy, dark = low energy)
ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "energy_blue",
    ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#f3f8fe"],
)
# diverging map for final m_z (blue = +z, red = -z, grey midpoint)
MZ_CMAP = LinearSegmentedColormap.from_list(
    "mz_div", ["#a32b2b", "#e34948", "#f2a5a4", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"]
)


def apply_style(base_size: float = 8.0) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": base_size,
        "axes.titlesize": base_size + 0.5,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "legend.fontsize": base_size - 0.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "legend.frameon": False,
        "lines.linewidth": 1.3,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })


def save(fig, path: str | Path, pdf: bool = True) -> None:
    """Save at the exact figure size (no tight bbox) so aspect ratios are preserved."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# time-domain plots
# ----------------------------------------------------------------------
def plot_dynamics(res: SimResult, figsize=(6.0, 3.6), title: str | None = None,
                  components=("mx", "my", "mz"), legend_loc="upper right"):
    """Magnetization components (top) and the applied voltage (bottom)."""
    fig, (ax, axv) = plt.subplots(2, 1, figsize=figsize, sharex=True, layout="constrained",
                                  gridspec_kw={"height_ratios": [3, 1]})
    labels = {"mx": r"$m_x$", "my": r"$m_y$", "mz": r"$m_z$"}
    for c in components:
        ax.plot(res.t_ns, getattr(res, c), color=COMPONENT_COLORS[c], label=labels[c],
                lw=1.6 if c == "mz" else 1.0)
    ax.axhline(0, color=MUTED, lw=0.6, ls="--")
    ax.set_ylim(-1.08, 1.08)
    ax.set_ylabel("Magnetization")
    ax.legend(loc=legend_loc, ncol=3, handlelength=1.6, columnspacing=1.0)
    if title:
        ax.set_title(title, loc="left")
    axv.plot(res.t_ns, res.V, color=INK2, lw=1.2)
    axv.set_ylabel("V (V)")
    axv.set_xlabel("Time (ns)")
    vmax = max(1.0, float(np.max(res.V)))
    axv.set_ylim(-0.08 * vmax, 1.25 * vmax)
    axv.set_xlim(res.t_ns[0], res.t_ns[-1])
    return fig, (ax, axv)


# ----------------------------------------------------------------------
# energy landscapes
# ----------------------------------------------------------------------
def landscape_2d(ax, device: Device, V: float, bias=(0.0, 0.0, 0.0), res: SimResult | None = None,
                 n=181, levels=20, vlim=None, label_lines=False):
    """Energy density (kJ/m^3) over polar angle theta (x) and azimuth phi (y)."""
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(-np.pi / 2, np.pi / 2, n)
    TH, PH = np.meshgrid(th, ph)
    E = energy_map(device, TH, PH, V, bias) / 1e3
    kw = {} if vlim is None else {"vmin": vlim[0], "vmax": vlim[1]}
    cf = ax.contourf(np.degrees(th), np.degrees(ph), E, levels=levels, cmap=ENERGY_CMAP, **kw)
    cl = ax.contour(np.degrees(th), np.degrees(ph), E, levels=8, colors="white", linewidths=0.35, alpha=0.7,
                    negative_linestyles="solid")
    if label_lines:
        ax.clabel(cl, inline=True, fontsize=5, fmt="%.0f")
    if res is not None:
        m = res.m
        theta = np.degrees(np.arccos(np.clip(m[:, 2], -1, 1)))
        phi = np.degrees(np.arctan2(m[:, 1], m[:, 0]))
        # break the line where phi wraps outside the plotted window
        phi = np.where(np.abs(phi) > 90, np.nan, phi)
        ax.plot(theta, phi, color=TRAJ, lw=0.6)
        ax.plot(theta[0], phi[0], "o", ms=3.5, mfc="white", mec=INK, mew=0.6, zorder=5)
        ax.plot(theta[-1], phi[-1], "o", ms=3.5, mfc=INK, mec="white", mew=0.6, zorder=5)
    ax.set_xlim(0, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks([0, 90, 180])
    ax.set_yticks([-90, 0, 90])
    ax.grid(False)
    return cf


def sphere(ax, device: Device, V: float, bias=(0.0, 0.0, 0.0), res: SimResult | None = None,
           n=160, bands=12, elev=30, azim=-60):
    """Banded energy landscape on the unit sphere, optionally with a trajectory."""
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, 2 * n)
    PH, TH = np.meshgrid(ph, th)
    M = unit_vector(TH, PH)
    E = energy_density(device, M, V, bias)
    En = (E - E.min()) / (E.max() - E.min() + 1e-30)
    colors = ENERGY_CMAP(np.floor(En * bands) / bands)
    ax.plot_surface(M[..., 0], M[..., 1], M[..., 2], facecolors=colors, rstride=1, cstride=1,
                    antialiased=False, shade=False, linewidth=0, alpha=1.0)
    if res is not None:
        m = res.m * 1.01
        # draw only the visible (front) part to avoid see-through clutter
        view = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                         np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                         np.sin(np.radians(elev))])
        vis = (res.m @ view) > -0.05
        xs, ys, zs = (np.where(vis, m[:, k], np.nan) for k in range(3))
        ax.plot(xs, ys, zs, color=TRAJ, lw=0.6, zorder=10)
        ax.scatter(*m[0], s=10, c="white", edgecolors=INK, linewidths=0.6, zorder=11, depthshade=False)
        ax.scatter(*m[-1], s=10, c=INK, edgecolors="white", linewidths=0.6, zorder=11, depthshade=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    lim = 0.72
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.axis("off")


def trajectory_3d(ax, res: SimResult, elev=22, azim=-55):
    """Trajectory of m on a wire-frame unit sphere."""
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                      np.outer(np.ones_like(u), np.cos(v)), color=GRID, linewidth=0.4)
    ax.plot(res.mx, res.my, res.mz, color=TRAJ, lw=0.7)
    ax.scatter(*res.m[0], s=18, c="white", edgecolors=INK, linewidths=0.8, depthshade=False, label="start")
    ax.scatter(*res.m[-1], s=18, c=INK, edgecolors="white", linewidths=0.8, depthshade=False, label="end")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(r"$m_x$", labelpad=-8)
    ax.set_ylabel(r"$m_y$", labelpad=-8)
    ax.set_zlabel(r"$m_z$", labelpad=-8)
    ax.tick_params(pad=-3, labelsize=6)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticks([-1, 0, 1])
        a.pane.fill = False
        a.pane.set_edgecolor(GRID)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", fontsize=6, handletextpad=0.2, borderaxespad=0)


def mz_norm(vmax=1.0):
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
