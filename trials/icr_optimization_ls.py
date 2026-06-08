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


# -----------------------------------------------------------------------------
# Error function — time-aligned residuals on the experimental grid
# -----------------------------------------------------------------------------
def error_function_general(sim_params, id, erf_id,
                           data, dt, theta_interpolator, w_interpolator,
                           direction, optype='residual_xy',
                           sim_order=1, fw_type='interpolate',
                           w_orbit=1.0):
    time, X, Y, T, wf, rx, ry, Rorbit, etaf = data
    i0, i1 = id
    x0, y0, phi_0 = X[i0], Y[i0], T[i0]
    t_0, t_f = time[i0], time[i1]
    sim_time = t_f - t_0

    fv1, fv2 = erfs[erf_id](sim_params, theta_interpolator, w_interpolator, direction)

    if sim_order == 1:
        b0 = Ell2D(r=np.array([x0, y0]), theta_rad=phi_0, t0=t_0,
                   vfunc1=fv1, vfunc2=fv2, wfunc=w_interpolator)
        df_sim = b0.simulate(sim_time, dt, 'leg')
    elif sim_order == 2:
        b0 = BBot(r=np.array([x0, y0]), theta_rad=phi_0, t0=t_0,
                  vfunc1=fv1, vfunc2=fv2, wfunc=w_interpolator)
        if fw_type == 'interpolate':
            df_sim = simulate_bbot(b0, sim_time, dt, 'CN', 'time')
        elif fw_type == 'fft':
            df_sim = simulate_bbot(b0, sim_time, dt, 'CN', 'angle')
        else:
            raise ValueError(f"unknown fw_type={fw_type!r}")
    else:
        raise ValueError(f"unknown sim_order={sim_order!r}")

    # Time-align the simulation onto the experimental grid for [i0:i1]
    t_sim = df_sim['time'].values if 'time' in df_sim.columns else df_sim.index.values
    t_q = np.asarray(time[i0:i1])

    def interp(arr):
        arr = np.asarray(arr)
        return interp1d(t_sim, arr, bounds_error=False,
                        fill_value=(arr[0], arr[-1]))(t_q)

    sim_X    = interp(df_sim.X.values)
    sim_Y    = interp(df_sim.Y.values)
    sim_Gicrx = interp(df_sim.Gicrx.values)
    sim_Gicry = interp(df_sim.Gicry.values)
    sim_ro    = interp(np.abs(df_sim['vpmag'].values / df_sim['w'].values))

    residual_x  = X[i0:i1]      - sim_X
    residual_y  = Y[i0:i1]      - sim_Y
    residual_rx = rx[i0:i1]     - sim_Gicrx
    residual_ry = ry[i0:i1]     - sim_Gicry
    residual_ro = (Rorbit[i0:i1] - sim_ro) * w_orbit

    if optype == 'residual_xy':
        return np.concatenate([residual_x, residual_y])
    elif optype == 'residual_icr':
        return np.concatenate([residual_rx, residual_ry, residual_ro])
    elif optype == 'euclidian':
        return float(np.sum(np.sqrt(residual_x ** 2 + residual_y ** 2)))
    else:
        raise ValueError(f"unknown optype={optype!r}")


# -----------------------------------------------------------------------------
# Per-case optimizer
# -----------------------------------------------------------------------------
def optimize_ls(i, j, k, erf_id,
                normalize=False, optype='residual_xy',
                fw_type='interpolate', sim_order=1,
                w_orbit=1.0):
    edt = dtime[i][j][k]
    time = dfs[i][j][k].Time
    df = dfs[i][j][k]
    dirr = np.sign(np.average(df.w))
    Xf, Yf, Thetaf, vpxf, vpyf, wf = filter_XY(df, edt, True, 21, 3)

    # Instantaneous-center quantities used as fit targets
    rx     = Xf - vpyf / wf
    ry     = Yf + vpxf / wf
    Rorbit = np.sqrt(vpxf ** 2 + vpyf ** 2) / np.abs(wf)
    etaf   = savgol_filter(np.abs(wf)/ df.vmag, 21, 3) * 1.45 / 100

    data = [time, Xf, Yf, Thetaf, wf, rx, ry, Rorbit, etaf]

    l0 = 0
    l1 = len(df) - 1
    sim_time = time[l1] - time[l0]

    # Interpolators for theta and w
    if fw_type == 'interpolate':
        fw_interp    = interp1d(time, wf,     kind='cubic', fill_value="extrapolate")
        theta_interp = interp1d(time, Thetaf, kind='cubic', fill_value="extrapolate")
        fw    = lambda t: float(np.asarray(fw_interp(t)))
        theta_func = lambda t: float(np.asarray(theta_interp(t)))
    elif fw_type == 'fft':
        raise NotImplementedError("fw_type='fft' branch not wired up here")
    else:
        raise ValueError(f"unknown fw_type={fw_type!r}")

    # Residual closure
    error_compute = lambda params: error_function_general(
        params, [l0, l1], erf_id, data, edt,
        theta_interp, fw, dirr, optype, sim_order, fw_type, w_orbit
    )

    # Pick optimizer: any 'residual_*' goes to least_squares
    x_init = np.random.rand(erf_param_len[erf_id])
    if optype.startswith('residual'):
        result = least_squares(error_compute, x_init)
        final = result.cost
    elif optype == 'euclidian':
        result = minimize(error_compute, x_init, method='L-BFGS-B')
        final = result.fun
    else:
        raise ValueError(f"unknown optype={optype!r}")

    optimized_params = result.x
    fv1, fv2 = erfs[erf_id](optimized_params, theta_interp, fw_interp, dirr)
    print(optimized_params, np.round(final, 5))

    vs = [optimized_params]

    # Simulate with optimal params
    if sim_order == 1:
        b0 = Ell2D(r=[Xf[l0], Yf[l0]], theta_rad=Thetaf[l0], t0=time[l0],
                   vfunc1=fv1, vfunc2=fv2, wfunc=fw)
        df_sim = b0.simulate(sim_time, edt, 'leg')
    elif sim_order == 2:
        b0 = BBot(r=[Xf[l0], Yf[l0]], theta_rad=Thetaf[l0], t0=time[l0],
                  vfunc1=fv1, vfunc2=fv2, wfunc=fw)
        if fw_type == 'interpolate':
            df_sim = simulate_bbot(b0, sim_time, edt, 'CN', 'time')
        elif fw_type == 'fft':
            df_sim = simulate_bbot(b0, sim_time, edt, 'CN', 'angle')
            b0.df = df_sim
            b0.postprocess()
    else:
        raise ValueError(f"unknown sim_order={sim_order!r}")

    return data, df_sim, result, vs


def compute_least_squares_cost(X, Y, Xfit, Yfit):
    residual_x = X - Xfit
    residual_y = Y - Yfit
    residuals = np.concatenate([residual_x, residual_y])
    cost_i = np.sqrt(residual_x ** 2 + residual_y ** 2)
    cost = 0.5 * np.sum(residuals ** 2)
    return cost_i, cost


# -----------------------------------------------------------------------------
# Run the four cases
# -----------------------------------------------------------------------------
edatas  = []
opts    = []
ress    = []
vparams = []

ijk    = [[2, 0, 8], [4, 1, 5], [0, 0, 5], [2, 1, 7]]
models = [11, 11, 11, 11]

PARAM_DIR = '../outputs/data'
os.makedirs(PARAM_DIR, exist_ok=True)

for c in range(len(ijk)):
    i, j, k = ijk[c]
    edata, opt8, res, vs = optimize_ls(
        i, j, k, models[c],
        optype='residual_icr', fw_type='interpolate', sim_order=2
    )
    edatas.append(edata)
    opts.append(opt8)
    ress.append(res)
    vparams.append(vs)

# -----------------------------------------------------------------------------
# Write optimized parameters
# -----------------------------------------------------------------------------
# Per-case .npz (full fidelity: params, cost, success, ijk, model id)
for c, (case_ijk, res) in enumerate(zip(ijk, ress)):
    i, j, k = case_ijk
    np.savez(
        os.path.join(PARAM_DIR, f'params_case{c}_i{i}_j{j}_k{k}.npz'),
        params=res.x,
        cost=float(getattr(res, 'cost', getattr(res, 'fun', np.nan))),
        success=bool(getattr(res, 'success', True)),
        ijk=np.array(case_ijk),
        erf_id=models[c],
    )

# Combined CSV (one row per case, columns p0..pN)
n_params = max(len(r.x) for r in ress)
cols = ['case', 'i', 'j', 'k', 'erf_id', 'cost'] + [f'p{n}' for n in range(n_params)]
rows = []
for c, (case_ijk, res) in enumerate(zip(ijk, ress)):
    i, j, k = case_ijk
    row = {
        'case': c, 'i': i, 'j': j, 'k': k,
        'erf_id': models[c],
        'cost': float(getattr(res, 'cost', getattr(res, 'fun', np.nan))),
    }
    for n in range(n_params):
        row[f'p{n}'] = res.x[n] if n < len(res.x) else np.nan
    rows.append(row)

params_csv = os.path.join(PARAM_DIR, 'params_all_cases.csv')
pd.DataFrame(rows, columns=cols).to_csv(params_csv, index=False)
print(f"Wrote parameters: {params_csv} (+ {len(ijk)} per-case .npz files in {PARAM_DIR})")

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(10, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

letters = 'abcdefghijkl'
shift = 0
shift_delta = lambda x: 1.01 * (np.max(x) - np.min(x))

for c in range(len(ijk)):
    i, j, k = ijk[c]
    df    = dfs[i][j][k]
    edata = edatas[c]
    opt8  = opts[c]

    ax_top    = axs[0, c]
    ax_bottom = axs[1, c]

    # Top row: trajectory
    ax_top.plot(edata[1] * 100, edata[2] * 100, c='black', lw=1)
    ax_top.plot((opt8.X + shift_delta(opt8.X) * shift) * 100, opt8.Y * 100, c='C3')
    ax_top.set_xlabel('x (cm)')
    if c == 0:
        ax_top.set_ylabel('y (cm)')
    ax_top.axis('equal')
    ax_top.text(0.85, 0.9, f'({letters[c]})', transform=ax_top.transAxes)

    # Bottom row: eta(t) — edata[8] is etaf, not edata[7] (Rorbit)
    ax_bottom.plot(edata[0], edata[8], c='black', lw=1)
    ax_bottom.plot(opt8.time, opt8.sigma, c='C3')
    ax_bottom.set_xlabel('t (s)')
    if c == 0:
        ax_bottom.set_ylabel(r'$\eta$')
    ax_bottom.text(0.866, 0.9, f'({letters[c + 4]})', transform=ax_bottom.transAxes)
    ax_bottom.set_ylim([-0.1, 2])

fig.savefig(f'../code_outputs/figures/constopt_4series2_shifted{shift}_.pdf', dpi=600)
plt.show()