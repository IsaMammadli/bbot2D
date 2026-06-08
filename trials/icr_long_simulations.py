import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
import random

print(os.getcwd())

# user defined  (wildcard imports first — they may pollute the namespace)
from src.Ellipse2dObj import *
from src.animate import *
from src.trajectory import *
from src.simopt import *
from src.optimizer_utils import *
from src import bbotData

# scipy imports AFTER wildcard imports so `norm`, `skewnorm`, `interp1d`
# are not shadowed (e.g. by numpy.linalg.norm leaking via `from numpy import *`)
from scipy.interpolate import interp1d
from scipy.stats import norm as scipy_norm, skewnorm
norm = scipy_norm   # explicit rebinding, defensive against later shadowing

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


# -----------------------------------------------------------------------------
# Cases: must match what was fitted/saved by the optimization script
# -----------------------------------------------------------------------------
ijk          = [[2, 0, 8], [4, 1, 5], [0, 0, 5], [2, 1, 7]]
trajectories = [dfs[i][j][k]   for (i, j, k) in ijk]
dts          = [dtime[i][j][k] for (i, j, k) in ijk]

# -----------------------------------------------------------------------------
# Load fitted parameters from ../outputs/data/
# -----------------------------------------------------------------------------
PARAM_DIR = '../outputs/data'
EXPECTED_N_PARAMS = 6   # [vo_1, vo_2, vh_1, vh_2, vh_3, vh_4]

def load_case_params(c, i, j, k, param_dir=PARAM_DIR):
    fname = os.path.join(param_dir, f'params_case{c}_i{i}_j{j}_k{k}.npz')
    if not os.path.isfile(fname):
        raise FileNotFoundError(
            f"Missing fitted-params file: {fname}\n"
            f"Run the optimization script first so it writes into {param_dir}."
        )
    with np.load(fname) as d:
        p    = np.asarray(d['params'])
        cost = float(d['cost']) if 'cost' in d.files else np.nan
    if p.size != EXPECTED_N_PARAMS:
        raise ValueError(
            f"{fname}: expected {EXPECTED_N_PARAMS} params, got {p.size}. "
            f"Did the erf model change?"
        )
    return p, cost

params = []
for c, (i, j, k) in enumerate(ijk):
    p, cost = load_case_params(c, i, j, k)
    params.append(p)
    print(f"case {c} (i={i}, j={j}, k={k}) cost={cost:.5g}  params={p}")

thresholds = [[0.1,  0.0385, 0.035, 0.0  ],
              [0.1,  0.05,   0.01,  0.001],
              [0.2,  0.15,   0.05,  0.01 ],
              [0.1,  0.02,   0.01,  0.001]]

thresholds_f = [[2.5, 2.5, 2.5, 2.5]] * 4


# -----------------------------------------------------------------------------
# Outlier filtering helper
# -----------------------------------------------------------------------------
def remove_outliers(data, method="iqr", k=1.5):
    if method == "std":
        mu, std = np.mean(data), np.std(data)
        return data[(data > mu - k * std) & (data < mu + k * std)]
    elif method == "iqr":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        return data[(data >= lower) & (data <= upper)]
    elif method == "percentile":
        lower, upper = np.percentile(data, [k, 100 - k])
        return data[(data >= lower) & (data <= upper)]
    else:
        return data  # no filtering


# -----------------------------------------------------------------------------
# Fit a skew-normal to each case's filtered angular velocity
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 4))
normal_dists = []
skewed_dists = []

for i in range(4):
    Xf, Yf, Thetaf, vpxf, vpyf, wf = filter_XY(trajectories[i], dts[i], True, 7, 2)
    w0 = wf
    w0_clean = remove_outliers(w0, method="none", k=1.5)
    a, loc, scale = skewnorm.fit(w0_clean)
    mu, std = norm.fit(w0_clean)
    normal_dists.append(norm(mu, std))
    skewed_dists.append(skewnorm(a, loc, scale))


# -----------------------------------------------------------------------------
# Monte-Carlo simulation + figure
# -----------------------------------------------------------------------------
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10, 14))
gs = gridspec.GridSpec(6, 4, hspace=0.45, wspace=0.3)
plot_type = 0

for tr_id in range(4):
    df = trajectories[tr_id]
    edt = dts[tr_id]
    vo_1, vo_2, vh_1, vh_2, vh_3, vh_4 = params[tr_id]
    Xf, Yf, Thetaf, vpxf, vpyf, wf = filter_XY(df, edt, True, 7, 2)
    vpmag = np.sqrt(vpxf ** 2 + vpyf ** 2)
    tt = df.Time
    dt = 0.03
    T = np.max(df.Time) * 10

    accumulated_w     = []
    accumulated_vpmag = []
    accumulated_sigma = []

    for cnt in range(10):
        np.random.seed(cnt + 100)
        N = int(T / dt) + 1
        random_w = skewed_dists[tr_id].rvs(size=N)
        fw = lambda t: random_w[int(t / dt)] if t < T else 0

        fv1 = lambda t: vo_1 + vh_1 * np.cos(t) - vh_2 * np.sin(t)
        fv2 = lambda t: vo_2 - vh_3 * np.cos(t) - vh_4 * np.sin(t)
        b0 = BBot(r=np.array([Xf[0], Yf[0]]), theta_rad=Thetaf[0],
                  vfunc1=fv1, vfunc2=fv2, wfunc=fw)

        df3 = simulate_bbot(b0, T, dt, 'CN', dependency='angle')
        b0.df = df3
        b0.postprocess()

        accumulated_w.extend(random_w[:len(df3)])
        accumulated_vpmag.extend(df3.vpmag)
        accumulated_sigma.extend(df3.sigma)

        if cnt == 0:
            ax0      = fig.add_subplot(gs[0, tr_id])   # Row 0: short-time traj
            ax1      = fig.add_subplot(gs[1, tr_id])   # Row 1: full traj
            ax3      = fig.add_subplot(gs[2, tr_id])   # Row 2: ω(t)
            ax2      = fig.add_subplot(gs[3, tr_id])   # Row 3: P(ω)
            ax5      = fig.add_subplot(gs[4, tr_id])   # Row 4: P(v)
            ax4      = fig.add_subplot(gs[5, tr_id])   # Row 5: P(η)
            ax_inset = inset_axes(ax3, width="45%", height="40%", loc=3, borderpad=-1.4)

        ax0.plot(df3.X[:int(23 / dt)] * 100, df3.Y[:int(23 / dt)] * 100, lw=1)
        ax1.plot(df3.X * 100, df3.Y * 100, lw=1)

    ax0.axis('equal')
    ax1.axis('equal')

    # Row 2: angular velocity time series
    ax3.plot(tt, wf, label='Original ω', c='black')
    ax3.plot(df3.time, random_w[:len(df3)], label='Filtered ω', c='C3', alpha=0.6)

    # Inset on ax3
    ax_inset.plot(tt, wf, c='black', lw=0.5)
    ax_inset.plot(df3.time[:int(len(df))], random_w[:int(len(df))], c='C3', alpha=0.5, lw=0.5)
    ax_inset.tick_params(axis='both', which='both', labelsize=6)

    if plot_type == 0:
        wrange = np.linspace(np.min(wf), np.max(wf), 200)

        # Row 3: P(ω) - ensemble-averaged
        ax2.hist(wf, bins=60, density=True, color="black")
        ax2.hist(accumulated_w, bins=60, density=True, color="C3", alpha=0.)
        ax2.plot(wrange, skewed_dists[tr_id].pdf(wrange), "r--", lw=2)

        # Row 4: P(v) - ensemble-averaged
        ax5.hist(vpmag, bins=60, density=True, color="black")
        ax5.hist(np.array(accumulated_vpmag), density=True, bins=60, color='C3', alpha=0.6)

        # Row 5: P(η) - ensemble-averaged
        ax4.hist(np.abs(wf / vpmag * 1.45 / 100), density=True, bins=60, color='black')
        ax4.hist(accumulated_sigma, density=True, bins=60, color='C3', alpha=0.6)
    else:
        ax2.plot(tt, wf, c='black')
        ax5.plot(tt, wf, c='black')
        ax4.plot(df.Time, np.abs(wf / vpmag * rmag), 'black')

    # x-labels
    ax1.set_xlabel("x (cm)")
    ax3.set_xlabel(r't (s)')
    ax2.set_xlabel(r"$\omega$ (rad/s)")
    ax5.set_xlabel(r"$v$ (m/s)")
    ax4.set_xlabel(r"$\eta$")

    # y-labels (leftmost column only)
    if tr_id == 0:
        ax0.set_ylabel("y (cm)")
        ax1.set_ylabel("y (cm)")
        ax3.set_ylabel(r"$\omega$ (rad/s)")
        ax2.set_ylabel(r"$P(\omega)$")
        ax5.set_ylabel(r"$P(v)$")
        ax4.set_ylabel(r"$P(\eta)$")

plt.savefig(f'../code_outputs/figures/multiple_noise_inset_{tr_id}_icr.pdf', dpi=300)
plt.show()