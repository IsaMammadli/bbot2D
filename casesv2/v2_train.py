import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

from scipy.interpolate import interp1d, CubicSpline
from scipy.stats import norm, skewnorm
from scipy.optimize import least_squares, minimize


os.makedirs(out_dir, exist_ok=True)

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

RESIDUAL_TYPE = 'xy' #icr

# ------------------------------------------------------------------ #
def error_function_general(sim_params, id, erf_id,
                           data, dt, theta_interpolator, w_interpolator,
                           direction, optype='residual'):
    time, X, Y, T, wf, Rx, Ry, etaf, Rorbit = data
    x0, y0, phi_0 = X[id[0]], Y[id[0]], T[id[0]]
    t_0 = time[id[0]]
    t_f = time[id[1]]
    sim_time = t_f - t_0
    fv1, fv2 = erfs[erf_id](sim_params, theta_interpolator, w_interpolator, direction)

    b0 = BBot(r=np.array([x0, y0]), theta_rad=phi_0, t0=t_0,
              vfunc1=fv1, vfunc2=fv2, wfunc=w_interpolator)
    df_sim = simulate_bbot(b0, sim_time, dt, 'CN', 'time')

    residual_x = (X[id[0]:id[1]] - df_sim.X)
    residual_y = (Y[id[0]:id[1]] - df_sim.Y)
    residual_rx = (Rx[id[0]:id[1]] - df_sim.Gicrx)
    residual_ry = (Ry[id[0]:id[1]] - df_sim.Gicry)
    sim_ro = df_sim.vpmag/np.abs(df_sim.w)
    residual_ro = (Rorbit[id[0]:id[1]] - sim_ro)
    if RESIDUAL_TYPE == 'icr':
        if optype == 'residual':
            return np.concatenate([residual_rx, residual_ry, residual_ro])
    elif RESIDUAL_TYPE=='xy':
            return np.concatenate([residual_x, residual_y])



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
    etaf   = np.abs(wf) / vpfmag * 1.45 / 100

    rx = Xf - vpyf / wf
    ry = Yf + vpxf / wf
    Rorbit = np.sqrt((vpyf / wf) ** 2 + (vpxf / wf) ** 2)

    data = [time, Xf, Yf, Thetaf, wf, rx, ry, etaf, Rorbit]
    return edt, data


def optimize_ls(l1, edt, data, erf_id, sim_time,
                optype='residual',
                init_scale=0.05,
                method='lm',
                max_nfev=300,
                xtol=1e-6, ftol=1e-6, gtol=1e-6,
                x0=None,
                verbose=1):
    """
    Fit on samples [0, l1) of `data`, then simulate for duration `sim_time`.

    Speed knobs:
      init_scale : random init is U(0,1) * init_scale  (physical vh_* are ~ 0.01-0.2)
      method     : 'lm' (faster, no bounds) or 'trf'
      max_nfev   : hard cap on residual evaluations
      x0         : pass an explicit initial guess (warm start) instead of random
    """
    time, Xf, Yf, Thetaf, wf, rx, ry, etaf, Rorbit = data
    dirr = np.sign(np.average((wf)))
    l1 = min(int(l1), len(time) - 1)
    lw = l1

    # CubicSpline is a compiled BSpline evaluator. For scalar t inside the
    # simulation loop it is several times faster than interp1d(kind='cubic')
    # followed by .item(). interp1d's __call__ is mostly Python overhead.
    fw_cs    = CubicSpline(time[:lw], wf[:lw],     extrapolate=True)
    theta_cs = CubicSpline(time[:lw], Thetaf[:lw], extrapolate=True)
    fw           = lambda t: float(fw_cs(t))
    fw_interp    = fw_cs
    theta_interp = theta_cs

    error_compute = lambda p: error_function_general(
        p, [0, l1], erf_id, data, edt, theta_interp, fw, dirr, optype)

    n_params = erf_param_len[erf_id]
    if x0 is None:
        x0 = init_scale * np.random.rand(n_params)

    if optype == 'residual':
        kwargs = dict(method=method, max_nfev=max_nfev,
                      xtol=xtol, ftol=ftol, gtol=gtol, verbose=verbose)
        if method == 'trf':
            kwargs['x_scale']   = 'jac'
            kwargs['diff_step'] = 1e-4
        result = least_squares(error_compute, x0, **kwargs)
    elif optype == 'euclidian':
        result = minimize(error_compute, x0, method='L-BFGS-B',
                          options=dict(maxiter=max_nfev))

    fv1, fv2 = erfs[erf_id](result.x, theta_interp, fw_interp, dirr)
    cost_print = result.fun if optype == 'euclidian' else result.cost
    print(f'  params: {result.x}\n  cost: {np.round(cost_print, 5)}  '
          f'nfev: {result.nfev}')

    b0 = BBot(r=[Xf[0], Yf[0]], theta_rad=Thetaf[0], t0=time[0],
              vfunc1=fv1, vfunc2=fv2, wfunc=fw)
    df_sim = simulate_bbot(b0, sim_time, edt, 'CN', 'time')
    return data, df_sim, result, [result.x]


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
tr1 = np.load('../bbotv2_data/mode_1.npy').T
tr1 = tr1[120+300:, :]
tr1 = tr1 - tr1[0, :]
time1, x1, y1, theta1 = tr1[:, :].T
x1 *= 1 / 100
y1 *= 1 / 100
theta1 = np.unwrap(theta1, period=np.pi) + np.pi * 1
edt1, dataf1 = filter_data(time1, x1, y1, theta1)

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


# ------------------------------------------------------------------ #
# Fit on percentages of data, then plot per trajectory
# ------------------------------------------------------------------ #
percentages = [100, 20, 10, 5]
letters     = 'abcdefghijklmnop'
scf         = 100  # m -> cm

n_tr  = 2
n_pct = len(percentages)
MODEL_ID = 15
N_PARAMS = erf_param_len[MODEL_ID]

all_skewparams = np.zeros((n_tr, n_pct, 3))
all_costs      = np.zeros((n_tr, n_pct))
all_params     = np.zeros((n_tr, n_pct, N_PARAMS))
NBINS = 100
for tr_id in range(n_tr):
    time, Xf, Yf, Thetaf, wf, rx, ry, etaf, Rorbit = datafs[tr_id]
    edt = edts[tr_id]

    # Warm-start within a trajectory: first fit picks a random init,
    # subsequent fits start from the previous solution.
    warm_x0 = None

    edatas, opts, ress, vparams = [], [], [], []
    for pct_idx, pct in enumerate(percentages):
        l1 = max(int(len(time) * pct / 100), 10)
        sim_time_pct = time[l1 - 1] - time[0]

        print(f'tr{tr_id}  pct={pct:3d}%  l1={l1}')
        edata, opt, res, vs = optimize_ls(
            l1, edt, datafs[tr_id], MODEL_ID, sim_time_pct,
            optype='residual',
            init_scale=0.05,
            method='lm',
            max_nfev=300,
            x0=warm_x0,
        )
        warm_x0 = res.x.copy()

        edatas.append(edata); opts.append(opt); ress.append(res); vparams.append(vs)
        all_params[tr_id, pct_idx] = res.x
        all_costs[tr_id,  pct_idx] = res.cost

        a, loc, scale = skewnorm.fit(wf[:l1])
        all_skewparams[tr_id, pct_idx] = [a, loc, scale]

    # ------ figure: 3 rows x n_pct columns -------------------------- #
    fig, axs = plt.subplots(3, n_pct, figsize=(3.5 * n_pct, 8))
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    for col, pct in enumerate(percentages):
        edata = edatas[col]
        opt   = opts[col]
        l1    = max(int(len(time) * pct / 100), 10)

        ax0 = axs[0, col]
        ax1 = axs[1, col]
        ax2 = axs[2, col]

        ax0.plot(edata[1] * scf, edata[2] * scf, c='black', lw=1, label='data')
        ax0.plot(opt.X * scf, opt.Y * scf, c='C3', lw=1, label='sim')
        ax0.plot(edata[1][:l1] * scf, edata[2][:l1] * scf,
                 c='C0', lw=1.3, alpha=0.7, label=f'fit window ({pct}%)')
        ax0.set_xlabel('x (cm)')
        ax0.axis('equal')
        ax0.text(0.85, 0.9, f'({letters[col]})', transform=ax0.transAxes)
        ax0.set_title(f'fit on {pct}% of data', fontsize=10)
        if col == 0:
            ax0.set_ylabel('y (cm)')
            ax0.legend(fontsize=7, loc='best')

        a, loc, scale = all_skewparams[tr_id, col]
        w_slice = wf[:l1]
        ax1.hist(wf, density=True, color='black', bins=NBINS, alpha=0.55,
                 label='full $\\omega$')
        ax1.hist(w_slice, density=True, color='C0', bins=NBINS, alpha=0.45,
                 label=f'{pct}% slice')
        x_grid = np.linspace(wf.min(), wf.max(), 400)
        ax1.plot(x_grid, skewnorm.pdf(x_grid, a, loc, scale),
                 color='C3', linestyle='--', lw=1.5,
                 label=f'skewnorm\n$a$={a:.2f}\n$\\mu$={loc:.2f}\n$\\sigma$={scale:.2f}')
        ax1.set_xlabel(r'$\omega$ (rad/s)')
        ax1.text(0.85, 0.9, f'({letters[col + 4]})', transform=ax1.transAxes)
        ax1.set_ylim([0, 1.5])
        if col == 0:
            ax1.set_ylabel(r'$P(\omega)$')
        ax1.legend(fontsize=7, loc='best')

        ax2.hist(np.abs(edata[7]),    density=True, color='black',
                 bins=NBINS, alpha=0.7,  label='measured')
        ax2.hist(np.abs(opt.sigma[:]), density=True, color='C3',
                 bins=NBINS, alpha=0.55, label='sim')
        ax2.set_xlabel(r'$\eta$')
        ax2.text(0.85, 0.9, f'({letters[col + 8]})', transform=ax2.transAxes)
        ax2.set_ylim([0, 10])
        if col == 0:
            ax2.set_ylabel(r'$P(\eta)$')
            ax2.legend(fontsize=7, loc='best')

    fig.savefig(f'../code_outputs/figures/v2_train_trajectory_{tr_id}_{RESIDUAL_TYPE}.pdf', dpi=200)

np.savez(os.path.join(out_dir, 'fit_params.npz'),
         percentages=np.array(percentages),
         params=all_params,
         skewparams=all_skewparams,
         costs=all_costs,
         model_id=MODEL_ID,
         n_params=N_PARAMS)
print(f'saved → {os.path.join(out_dir, "fit_params.npz")}  (MODEL_ID={MODEL_ID}, N_PARAMS={N_PARAMS})')