import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
import random

from scipy.interpolate import interp1d
from scipy.stats import norm, skewnorm
from scipy.optimize import least_squares, minimize
from scipy.signal import savgol_filter

print(os.getcwd())

# user defined
from src.Ellipse2dObj import *
from src.animate import *
from src.trajectory import *
from src.simopt import *
from src.optimizer_utils import *
from src import bbotData

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
dfs, dtime, mps, la = bbotData.load()

# constants
A1 = 5.5 * 0.5 / 100
A2 = 3.0 * 0.5 / 100
rho1 = 0.374
rho2 = 0.661
rmag = np.sqrt((rho1 * A1) ** 2 + (rho2 * A2) ** 2)
Xmax = 53.4 / 100
Ymax = 48.4 / 100

DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "font.style": "italic",
})

tr1 = np.load('../bbotv2_data/mode_1.npy').T
tr1 = tr1[120:, :]
tr1 = tr1 - tr1[0, :]

time, x, y, theta = tr1[:, :].T
x *= 1 / 100
y *= 1 / 100
theta = np.unwrap(theta, period=np.pi) + np.pi

print('data len', len(x))

# -----------------------------------------------------------------------------
# Filter data
# -----------------------------------------------------------------------------
edt = np.average(np.diff(time))

win = 21
pol_degree = 3

Xf      = savgol_filter(x,     win, pol_degree, 0)
Yf      = savgol_filter(y,     win, pol_degree, 0)
vpxf    = savgol_filter(x,     win, pol_degree, 1) / edt
vpyf    = savgol_filter(y,     win, pol_degree, 1) / edt
vpfmag  = np.sqrt(vpxf ** 2 + vpyf ** 2)
wf      = savgol_filter(theta, win, pol_degree, 1) / edt
Thetaf  = savgol_filter(theta, win, pol_degree, 0)
etaf    = savgol_filter(wf / vpfmag, win, pol_degree) * 1.45 / 100
dirr    = np.sign(np.average(wf))

# Instantaneous-center quantities (used as fit targets)
rx     = Xf - vpyf / wf
ry     = Yf + vpxf / wf
Rorbit = np.sqrt((vpyf / wf) ** 2 + (vpxf / wf) ** 2)   # = |v_p| / |w|

data = [time, Xf, Yf, Thetaf, vpxf, vpyf, wf, rx, ry, etaf, Rorbit]


# -----------------------------------------------------------------------------
# Model builder + simulator  (w is an optimization parameter)
# -----------------------------------------------------------------------------
def build_and_simulate(params, t_end, dt, x0_pos, y0_pos, theta0):
    nu, vo_1, vo_2, vh_1, vh_2, vh_3, vh_4, w = params

    fw   = lambda t: w
    fcos = lambda t: np.cos(nu * t) * np.cos(fw(t) * t) - np.sin(nu * t) * np.sin(fw(t) * t)
    fsin = lambda t: np.sin(nu * t) * np.cos(fw(t) * t) + np.cos(nu * t) * np.sin(fw(t) * t)

    fv1 = lambda t: (vo_1 + vh_1 * fcos(t) - vh_2 * fsin(t))
    fv2 = lambda t: (vo_2 - vh_3 * fsin(t) - vh_4 * fcos(t))

    b = BBot(r=np.array([x0_pos, y0_pos]),
             theta_rad=theta0, t0=0,
             vfunc1=fv1, vfunc2=fv2, wfunc=fw)
    return simulate_bbot(b, t_end, dt, 'CN', 'time')


# -----------------------------------------------------------------------------
# Residuals (IC-frame: rx, ry, R_orbit), optionally restricted to time[:l1]
# -----------------------------------------------------------------------------
def residuals(params, data, edt, l1=None, w_orbit=1.0):
    time, Xf, Yf, Thetaf, vpxf, vpyf, wf, rx, ry, etaf, Rorbit = data
    if l1 is None or l1 > len(time):
        l1 = len(time)

    t_end = time[l1 - 1]
    df_sim = build_and_simulate(params, t_end, edt,
                                Xf[0], Yf[0], Thetaf[0])

    t_sim = df_sim['time'].values if 'time' in df_sim.columns else df_sim.index.values
    t_q   = time[:l1]

    def interp(arr):
        return interp1d(t_sim, arr, bounds_error=False,
                        fill_value=(arr[0], arr[-1]))(t_q)

    sim_rx = interp(df_sim.Gicrx.values)
    sim_ry = interp(df_sim.Gicry.values)
    # FIX: orbit radius is |v_p| / |w|, not sqrt(|v_p|/w)
    sim_ro = interp((df_sim.vpmag.values / np.abs(df_sim.w.values)))

    res_rx = rx[:l1]     - sim_rx
    res_ry = ry[:l1]     - sim_ry
    res_ro = (Rorbit[:l1] - sim_ro) * w_orbit

    return np.concatenate([res_rx, res_ry, res_ro])


# -----------------------------------------------------------------------------
# Initial guess and bounds (order: nu, vo_1, vo_2, vh_1, vh_2, vh_3, vh_4, w)
# -----------------------------------------------------------------------------
w0_guess = float(np.average(wf))
w_abs    = max(abs(w0_guess) * 3.0, 5.0)

param_names = ['nu', 'vo_1', 'vo_2', 'vh_1', 'vh_2', 'vh_3', 'vh_4', 'w']
x0 = np.array([-0.06,
                0.03,  0.03,
               -0.001, -0.001, -0.001, -0.001,
                w0_guess])
lb = np.array([-1.0,
               -0.5, -0.5,
               -0.5, -0.5, -0.5, -0.5,
               -w_abs])
ub = np.array([ 1.0,
                0.5,  0.5,
                0.5,  0.5,  0.5,  0.5,
                w_abs])

# -----------------------------------------------------------------------------
# Portion of data to fit
# -----------------------------------------------------------------------------
l1 = 2800
if l1 is None or l1 > len(time):
    l1 = len(time)

t_fit = time[:l1]
X_fit, Y_fit, Th_fit = Xf[:l1], Yf[:l1], Thetaf[:l1]

# Optional weight for the orbit-radius residual block. Set to e.g.
# 1.0 / np.mean(np.abs(Rorbit[:l1])) if you want it normalized.
w_orbit = 1.0

print(f"Fitting on {l1} / {len(time)} samples (t = {t_fit[0]:.3f} → {t_fit[-1]:.3f} s)")
print(f"Initial w guess = {w0_guess:.6f} rad/s   (bounds: ±{w_abs:.3f})")
print(f"Initial cost = {0.5 * np.sum(residuals(x0, data, edt, l1, w_orbit) ** 2):.4e}")

result = least_squares(
    residuals, x0,
    args=(data, edt, l1, w_orbit),
    bounds=(lb, ub),
    method='trf',
    x_scale='jac',
    diff_step=1e-4,
    xtol=1e-10, ftol=1e-10, gtol=1e-10,
    verbose=2,
)

print("\n--- Optimization result ---")
print(f"success     : {result.success}")
print(f"message     : {result.message}")
print(f"final cost  : {result.cost:.4e}")
print(f"nfev / njev : {result.nfev} / {result.njev}")
print("\nOptimal parameters:")
for name, val in zip(param_names, result.x):
    print(f"  {name:6s} = {val: .6f}")

nu_opt, vo_1, vo_2, vh_1, vh_2, vh_3, vh_4, w_opt = result.x
print(f"\nw_opt vs <wf> : {w_opt:+.4f}  vs  {w0_guess:+.4f}   (Δ = {w_opt - w0_guess:+.4e})")
print(f"(vh_1 - vh_2) = {vh_1 - vh_2:+.4e}")
print(f"(vh_3 - vh_4) = {vh_3 - vh_4:+.4e}")

# -----------------------------------------------------------------------------
# Simulate with optimal parameters over the FULL experimental window
# -----------------------------------------------------------------------------
df_opt = build_and_simulate(result.x, time[-1], edt,
                            Xf[0], Yf[0], Thetaf[0])
t_sim = df_opt['time'].values if 'time' in df_opt.columns else df_opt.index.values

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.plot(Xf[l1:], Yf[l1:], '-', color='lightgray', lw=1.5, label='experiment (not fitted)')
ax.plot(X_fit,   Y_fit,   '-', color='tab:blue',  lw=2.0, label=f'experiment (fitted, n={l1})')
ax.plot(df_opt.X, df_opt.Y, '--', color='tab:red', lw=1.8, label='simulation (optimal)')
ax.axvline(0, color='k', lw=0.3); ax.axhline(0, color='k', lw=0.3)
ax.set_xlabel(r'$X$ [m]'); ax.set_ylabel(r'$Y$ [m]')
ax.set_title('Trajectory'); ax.legend(); ax.set_aspect('equal'); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(time,  Xf,        '-',  color='tab:blue',  alpha=0.5, label=r'$X$ exp (all)')
ax.plot(time,  Yf,        '-',  color='tab:green', alpha=0.5, label=r'$Y$ exp (all)')
ax.plot(t_fit, X_fit,     '-',  color='tab:blue',  lw=1.8)
ax.plot(t_fit, Y_fit,     '-',  color='tab:green', lw=1.8)
ax.plot(t_sim, df_opt.X,  '--', color='tab:blue',  label=r'$X$ sim')
ax.plot(t_sim, df_opt.Y,  '--', color='tab:green', label=r'$Y$ sim')
ax.axvline(t_fit[-1], color='k', ls=':', lw=1.0, label='fit cutoff')
ax.set_xlabel(r'$t$ [s]'); ax.set_ylabel('position [m]')
ax.set_title('Position vs. time'); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../code_outputs/figures/trajectory_optimization.png', dpi=DPI, bbox_inches='tight')
plt.show()