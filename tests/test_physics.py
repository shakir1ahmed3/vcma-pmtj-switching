import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vcma_mtj import (MU0, Device, Pulse, PulseTrain, deng2017, demag_factors, energy_density,
                      equilibrium, simulate)
from vcma_mtj import experiments as ex


@pytest.fixture(scope="module")
def dev():
    return deng2017()


# ---------------------------------------------------------------- demag
def test_demag_reference_values():
    N = demag_factors(150e-9, 50e-9, 1.7e-9)
    assert N.Nz == pytest.approx(0.930077, abs=2e-6)
    assert N.Nx == pytest.approx(0.012241, abs=2e-6)
    assert N.Ny == pytest.approx(0.057682, abs=2e-6)
    assert sum(N) == pytest.approx(1.0, abs=1e-12)


def test_demag_limits():
    thin = demag_factors(150e-9, 50e-9, 0.05e-9)
    assert thin.Nz > 0.995  # infinite thin film -> Nz = 1
    circ = demag_factors(100e-9, 99.999e-9, 2e-9)
    assert circ.Nx == pytest.approx(circ.Ny, rel=1e-3)  # circular cross-section
    assert demag_factors(150e-9, 50e-9, 1.7e-9).Nx < demag_factors(150e-9, 50e-9, 1.7e-9).Ny


# ---------------------------------------------------------------- statics
def test_critical_voltages(dev):
    assert dev.V_xz == pytest.approx(4.744, abs=1e-3)
    assert dev.V_yz == pytest.approx(7.812, abs=1e-3)
    assert dev.V_crit_for_bias(50e-3) == pytest.approx(2.607, abs=1e-3)
    assert float(dev.Keff_xz(dev.V_xz)) == pytest.approx(0.0, abs=1e-6)
    assert 1e3 * float(dev.B_switch_z(0.0)) == pytest.approx(111.0, abs=0.1)


def test_half_period_eq8(dev):
    assert dev.period_bias_only(50e-3) / 2 == pytest.approx(0.357e-9, abs=1e-12)


def test_equilibrium_matches_analytic_cant(dev):
    Bx = 50e-3
    m = equilibrium(dev, 0.0, (Bx, 0, 0), +1)
    # for a field along the x axis of a biaxial particle: m_x = Bx / mu0 H_K,eff,xz
    assert m[0] == pytest.approx(Bx / float(dev.BKeff_xz(0.0)), abs=1e-5)
    m_lo = equilibrium(dev, 0.0, (Bx, 0, 0), -1)
    assert m_lo == pytest.approx(m * np.array([1, 1, -1]), abs=1e-5)


# ---------------------------------------------------------------- dynamics
def test_norm_and_energy_conservation(dev):
    undamped = dev.with_(alpha=0.0)
    m0 = np.array([np.sin(0.4), 0.0, np.cos(0.4)])
    r = simulate(undamped, PulseTrain(), m0, 2e-9, bias=(5e-3, 0, 0))
    assert np.allclose(np.linalg.norm(r.m, axis=1), 1.0, atol=1e-12)
    E = energy_density(undamped, r.m, 0.0, (5e-3, 0, 0))
    assert np.ptp(E) / abs(E[0]) < 1e-6


def test_rk4_matches_adaptive_solver(dev):
    pulses = PulseTrain(Pulse.from_t_on(1.6, 0.35e-9))
    r = simulate(dev, pulses, [0, 0, -1], 2e-9, bias=(50e-3, 0, 0), store_every=1)

    def rhs(t, m):
        V = float(pulses(t))
        B = np.array([-MU0 * dev.Ms * dev.Nx * m[0] + 50e-3,
                      -MU0 * dev.Ms * dev.Ny * m[1],
                      (2 * float(dev.Ku(V)) / dev.Ms - MU0 * dev.Ms * dev.Nz) * m[2]])
        mxB = np.cross(m, B)
        return -dev.gamma / (1 + dev.alpha**2) * (mxB + dev.alpha * np.cross(m, mxB))

    ref = solve_ivp(rhs, [0, 2e-9], [0, 0, -1], method="DOP853", rtol=1e-10, atol=1e-12,
                    max_step=1e-12, t_eval=[2e-9])
    assert r.final == pytest.approx(ref.y[:, -1], abs=1e-6)


def test_deng_reproduction(dev):
    runs = ex.run_deng(dev)
    assert runs[0.35e-9].switched and runs[1.05e-9].switched
    assert not runs[0.70e-9].switched


def test_near_critical_windows(dev):
    t = np.array([0.3e-9, 1.0e-9, 2.0e-9, 2.9e-9])
    mz = ex.switching_vs_width(dev, 4.8, t, (5e-3, 0, 0))
    assert list(mz < 0) == [False, True, False, True]


def test_pulse_shape():
    p = PulseTrain.single(2.0, 1e-9, t_start=0.1e-9, t_rise=0.1e-9, t_fall=0.1e-9)
    V = p(np.array([0.0, 0.15e-9, 0.5e-9, 1.25e-9, 2e-9]))
    assert V == pytest.approx([0.0, 1.0, 2.0, 1.0, 0.0])


def test_device_rejects_bad_demag():
    with pytest.raises(ValueError):
        Device(N=(0.1, 0.1, 0.1))
