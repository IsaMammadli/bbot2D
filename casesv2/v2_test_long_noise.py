import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from itertools import count
import random

print(os.getcwd())
# user defined
from src.Ellipse2dObj import *
from src.animate import *
from src.trajectory import *
from src.simopt import *
from src.optimizer_utils import *

# load data
from src import bbotData
dfs, dtime, mps, la = bbotData.load()

from scipy.interpolate import interp1d
from scipy.stats import norm, skewnorm
# from v2_train import RESIDUAL_TYPE

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
    "font.style": "italic"
})

RESIDUAL_TYPE = 'xy' #xy
# ------------------------------------------------------------------ #
# Load saved fit results
# ------------------------------------------------------------------ #
fit = np.load('../outputs/data/fit_params.npz')
percentages    = fit['percentages']
all_params     = fit['params']                  # (n_tr, n_pct, n_params)
all_skewparams = fit['skewparams']              # (n_tr, n_pct, 3)
all_costs      = fit['costs']                   # (n_tr, n_pct)
n_tr_saved     = all_params.shape[0]            # source of truth for trajectory count
print(f'loaded fit_params.npz: percentages={list(percentages)}, '
      f'params shape={all_params.shape}, n_tr={n_tr_saved}')


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def filter_data(time, x, y, theta):
    win = 21
    pol_degree = 3
    edt = np.average(np.diff(time))

    Xf     = savgol_filter(x,     win, pol_degree, 0)
    Yf     = savgol_filter(y,     win, pol_degree, 0)
    vpxf   = savgol_filter(x,     win, pol_degree, 1) / edt
    vpyf   = savgol_filter(y,     win, pol_degree, 1) / edt
    vpfmag = np.sqrt(vpxf ** 2 + vpyf ** 2)
    wf     = savgol_filter(theta, win, pol_degree, 1) / edt
    Thetaf = savgol_filter(theta, win, pol_degree, 0)
    etaf   = savgol_filter(wf / vpfmag, win, pol_degree) * 1.45 / 100

    rx = Xf - vpyf / wf
    ry = Yf + vpxf / wf
    Rorbit = np.sqrt((vpyf / wf) ** 2 + (vpxf / wf) ** 2)

    data = [time, Xf, Yf, Thetaf, vpxf, vpyf, wf, rx, ry, etaf, Rorbit]
    return edt, data


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
        return data


# ------------------------------------------------------------------ #
# Load trajectories
# ------------------------------------------------------------------ #
# mode 1
tr1 = np.load('../bbotv2_data/mode_1.npy').T
tr1 = tr1[120+300:, :]
tr1 = tr1 - tr1[0, :]
time1, x1, y1, theta1 = tr1[:, :].T
x1 *= 1 / 100
y1 *= 1 / 100
theta1 = np.unwrap(theta1, period=np.pi) + np.pi * 1
edt1, dataf1 = filter_data(time1, x1, y1, theta1)

# mode 2
tr2 = np.load('../bbotv2_data/mode_c.npy').T
tr2 = tr2[100:, :]
tr2 = tr2 - tr2[0, :]
time2, x2, y2, theta2 = tr2[:, :].T
x2 *= 1 / 100
y2 *= 1 / 100
theta2 = np.unwrap(theta2, period=np.pi) + np.pi * 0
edt2, dataf2 = filter_data(time2, x2, y2, theta2)

datafs = [dataf1, dataf2]
edts   = [edt1, edt2]

# Use only the trajectories that actually have saved fits, and guard
# against accidentally indexing past datafs.
n_tr = min(n_tr_saved, len(datafs))
if n_tr < n_tr_saved:
    print(f'warning: fit_params.npz has {n_tr_saved} trajectories but only '
          f'{len(datafs)} are loaded here; using n_tr={n_tr}')


# ------------------------------------------------------------------ #
# Main loop
# ------------------------------------------------------------------ #
n_ens     = 1
dt_sim    = 0.025
T_factor  = 2
plot_type = 0
MODEL_ID  = 15         # 11 or 15
NBINS     = 100
SIM_DEPENDENCY = 'timeAngle' if MODEL_ID == 15 else 'angle'

for tr_id in range(n_tr):
    n_pct = len(percentages)
    fig = plt.figure(figsize=(2.5 * n_pct, 14))
    gs  = gridspec.GridSpec(6, n_pct, hspace=0.45, wspace=0.3)

    time, Xf, Yf, Thetaf, vpxf, vpyf, wf, rx, ry, etaf, Rorbit = datafs[tr_id]
    edt = edts[tr_id]
    vpmag = np.sqrt(vpxf ** 2 + vpyf ** 2)
    T = np.max(time) * T_factor

    for col_id, pct in enumerate(percentages):
        pct = int(pct)
        if MODEL_ID == 11:
            vo_1, vo_2, vh_1, vh_2, vh_3, vh_4 = all_params[tr_id, col_id]
        elif MODEL_ID == 15:
            vo_1, vo_2, vh_1, vh_2, vh_3, vh_4, nu = all_params[tr_id, col_id]

        a, loc, scale = all_skewparams[tr_id, col_id]
        sk_dist = skewnorm(a, loc, scale)
        N_fit = int(pct * len(wf) / 100)
        print(f'tr{tr_id}  pct={pct:3d}%  N_fit={N_fit}  '
              f'a={a:+.3f}  loc={loc:+.4f}  scale={scale:.4f}  '
              f'cost={all_costs[tr_id, col_id]:.5f}')

        accumulated_w     = []
        accumulated_vpmag = []
        accumulated_sigma = []

        for cnt in range(n_ens):
            np.random.seed(cnt + 100)
            N = int(T / dt_sim) + 1
            random_w = sk_dist.rvs(size=N)

            fw = (lambda rw=random_w: (lambda t: rw[int(t/dt_sim)] if t < T else 0))()

            if MODEL_ID == 11:
                fv1 = (lambda v1=vo_1, h1=vh_1, h2=vh_2:
                    lambda t: v1 + h1 * np.cos(t) - h2 * np.sin(t))()
                fv2 = (lambda v2=vo_2, h3=vh_3, h4=vh_4:
                    lambda t: v2 - h3 * np.cos(t) - h4 * np.sin(t))()
            elif MODEL_ID == 15:
                fv1 = (lambda v1=vo_1, h1=vh_1, h2=vh_2, n=nu:
                    lambda t, th: v1 + h1 * np.cos(th + n*t) - h2 * np.sin(th + n*t))()
                fv2 = (lambda v2=vo_2, h3=vh_3, h4=vh_4, n=nu:
                    lambda t, th: v2 - h3 * np.cos(th + n*t) - h4 * np.sin(th + n*t))()

            b0 = BBot(r=np.array([Xf[0], Yf[0]]), theta_rad=Thetaf[0],
                      vfunc1=fv1, vfunc2=fv2, wfunc=fw)
            df3 = simulate_bbot(b0, T, dt_sim, 'CN', dependency=SIM_DEPENDENCY)
            b0.df = df3
            b0.postprocess()

            accumulated_w.extend(random_w[:len(df3)])
            accumulated_vpmag.extend(df3.vpmag)
            accumulated_sigma.extend(df3.sigma)

            if cnt == 0:
                ax0      = fig.add_subplot(gs[0, col_id])
                ax1      = fig.add_subplot(gs[1, col_id])
                ax3      = fig.add_subplot(gs[2, col_id])
                ax2      = fig.add_subplot(gs[3, col_id])
                ax5      = fig.add_subplot(gs[4, col_id])
                ax4      = fig.add_subplot(gs[5, col_id])
                ax_inset = inset_axes(ax3, width="45%", height="40%",
                                      loc=3, borderpad=-1.4)
                ax0.set_title(f'{pct}% of data // {N_fit} points', fontsize=9)

            ax0.plot(df3.X[:int(23/dt_sim)] * 100, df3.Y[:int(23/dt_sim)] * 100, lw=1)
            ax1.plot(df3.X * 100, df3.Y * 100, lw=1)

        ax0.axis('equal')
        ax1.axis('equal')

        ax3.plot(time, wf, c='black', lw=0.8)
        ax3.plot(time[:N_fit], wf[:N_fit], c='C0', lw=1.2, alpha=0.7)
        ax3.plot(df3.time, random_w[:len(df3)], c='C3', alpha=0.5, lw=0.7)

        ax_inset.plot(time, wf, c='black', lw=0.5)
        ax_inset.plot(df3.time[:len(time)], random_w[:len(time)],
                      c='C3', alpha=0.5, lw=0.5)
        ax_inset.tick_params(axis='both', which='both', labelsize=6)

        if plot_type == 0:
            wrange = np.linspace(np.min(wf), np.max(wf), 200)
            ax2.hist(wf, bins=NBINS, density=True, color="black")
            ax2.hist(accumulated_w, bins=NBINS, density=True, color="C3", alpha=0.4)
            ax2.plot(wrange, sk_dist.pdf(wrange), "r--", lw=2)
            ax5.hist(vpmag * 100, bins=NBINS, density=True, color="black")
            ax5.hist(np.array(accumulated_vpmag) * 100, bins=NBINS, density=True,
                     color='C3', alpha=0.6)
            ax4.hist(np.abs(wf / vpmag * 1.45 / 100), bins=NBINS, density=True, color='black')
            ax4.hist(accumulated_sigma, bins=NBINS, density=True, color='C3', alpha=0.6)
        else:
            ax2.plot(time, wf, c='black')
            ax5.plot(time, wf, c='black')
            ax4.plot(time, np.abs(wf / vpmag * rmag), 'black')

        ax1.set_xlabel("x (cm)")
        ax3.set_xlabel(r't (s)')
        ax2.set_xlabel(r"$\omega$ (rad/s)")
        ax5.set_xlabel(r"$v$ (cm/s)")
        ax4.set_xlabel(r"$\eta$")

        if col_id == 0:
            ax0.set_ylabel("y (cm)")
            ax1.set_ylabel("y (cm)")
            ax3.set_ylabel(r"$\omega$ (rad/s)")
            ax2.set_ylabel(r"$P(\omega)$")
            ax5.set_ylabel(r"$P(v)$")
            ax4.set_ylabel(r"$P(\eta)$")

        ax2.set_ylim([0, 1.5])
        ax5.set_ylim([0, 1.5])
        ax4.set_ylim([0, 10])

    fig.savefig(f'../code_outputs/figures/v2_test_noise_trajectory_{tr_id}_{RESIDUAL_TYPE}.pdf', dpi=300)
    # plt.show()