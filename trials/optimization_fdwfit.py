#This script runs optimization on long trajectory of v2 bot on different train (Ntrain) splits of the data.
#Fourier analysis is carried out on the experimental angular velocity data on these splits and considering time periodicity 
#longer trajectories are obtained.
import sys, os, pickle
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

print(os.getcwd())

# user defined
from src.Ellipse2dObj import *
from src.animate import *
from src.trajectory import *
from src.simopt import *
from src.optimizer_utils import *
# from fdwfit import *

# ---------------------------- constants ----------------------------
A1   = 5.5*0.5/100
A2   = 3.0*0.5/100
rho1 = 0.374
rho2 = 0.661
rmag = np.sqrt((rho1*A1)**2 + (rho2*A2)**2)
Xmax = 53.4/100
Ymax = 48.4/100

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size":   12,
    "font.style":  "italic",
})

# ---------------------------- make_phi -----------------------------
def make_phi(wf, time, n_modes, frac=1.0, anchor_t=None, anchor_value=None):
    """
    Fit the first `frac` of wf with the top-n_modes Fourier bins.
    Returns (wf_fn, phi_fn, info) — both callables accept scalar or array t.
        wf_fn(t)  : band-limited angular velocity
        phi_fn(t) : analytic integral, with phi_fn(anchor_t) = anchor_value
    """
    N_sub  = max(2, int(frac * len(wf)))
    wf_sub = wf[:N_sub]
    t_sub  = time[:N_sub]
    dt_sub = np.mean(np.diff(t_sub))
    t0_ref = time[0]

    if anchor_t     is None: anchor_t     = time[0]
    if anchor_value is None: anchor_value = 0.0

    W      = np.fft.rfft(wf_sub)
    amp    = np.abs(W)
    k_eff  = min(n_modes, len(amp))
    keep   = np.argpartition(amp, -k_eff)[-k_eff:]
    mask   = np.zeros_like(amp, dtype=bool); mask[keep] = True
    W_filt = W * mask

    freqs  = np.fft.rfftfreq(N_sub, d=dt_sub)
    omega  = 2*np.pi * freqs
    scales = np.full_like(freqs, 2.0)
    scales[0] = 1.0
    if N_sub % 2 == 0:
        scales[-1] = 1.0

    ac_freq   = omega[1:]
    ac_coef_w = scales[1:] * W_filt[1:] / N_sub
    ac_coef_p = ac_coef_w / (1j * ac_freq)
    dc_value  = W_filt[0].real / N_sub

    def wf_fn(t):
        t_arr      = np.asarray(t, dtype=float)
        was_scalar = (t_arr.ndim == 0)
        t_flat     = np.atleast_1d(t_arr).ravel()
        tau        = t_flat - t0_ref
        ac = (ac_coef_w[:, None]
              * np.exp(1j * ac_freq[:, None] * tau[None, :])
             ).sum(axis=0).real
        out = dc_value + ac
        if was_scalar:
            return float(out[0])
        return out.reshape(t_arr.shape)

    def F(t):
        t_arr      = np.asarray(t, dtype=float)
        was_scalar = (t_arr.ndim == 0)
        t_flat     = np.atleast_1d(t_arr).ravel()
        tau        = t_flat - t0_ref
        ac = (ac_coef_p[:, None]
              * np.exp(1j * ac_freq[:, None] * tau[None, :])
             ).sum(axis=0).real
        out = dc_value * tau + ac
        if was_scalar:
            return float(out[0])
        return out.reshape(t_arr.shape)

    F_anchor = F(anchor_t)              # now a Python float

    def phi_fn(t):
        return F(t) - F_anchor + anchor_value

    info = dict(
        n_sub        = N_sub,
        frac_used    = N_sub / len(wf),
        t_fit_end    = t_sub[-1],
        n_modes_kept = int(mask.sum()),
        period_sub   = N_sub * dt_sub,
    )
    return wf_fn, phi_fn, info


# ---------------------------- error fn -----------------------------
def error_function_general(sim_params, id, erf_id,
                           data, dt, theta_interpolator, w_interpolator,
                           direction, optype='residual'):
    time, X, Y, T, wf, Rx, Ry, etaf, Rorbit = data
    x0, y0, phi_0 = X[id[0]], Y[id[0]], T[id[0]]
    t_0    = time[id[0]]
    t_f    = time[id[1]]
    sim_time = t_f - t_0

    fv1, fv2 = erfs[erf_id](sim_params, theta_interpolator, w_interpolator, direction)

    b0 = BBot(r=np.array([x0, y0]), theta_rad=phi_0, t0=t_0,
              vfunc1=fv1, vfunc2=fv2, wfunc=w_interpolator)
    df_sim = simulate_bbot(b0, sim_time, dt, 'CN', 'time')

    # residual_x = X[id[0]:id[1]] - df_sim.X
    # residual_y = Y[id[0]:id[1]] - df_sim.Y
    res_rx  = Rx[id[0]:id[1]] - df_sim.Gicrx
    res_ry  = Ry[id[0]:id[1]] - df_sim.Gicry
    res_rb  = Rorbit[id[0]:id[1]] - np.sqrt(df_sim.vpmag/np.abs(df_sim.w))

    if optype == 'residual':
        # return np.concatenate([residual_x, residual_y])
        return np.concatenate([res_rx, res_ry, res_rb])
    elif optype == 'euclidian':
        # return np.sum(np.sqrt(residual_x**2 + residual_y**2))
        return np.sum(np.sqrt(res_rx**2 + res_ry**2+res_rb))

# ---------------------------- filtering ----------------------------
def filter_data(time, x, y, theta):
    win        = 21
    pol_degree = 3
    edt        = np.average(np.diff(time))

    Xf      = savgol_filter(x,     win, pol_degree, 0)
    Yf      = savgol_filter(y,     win, pol_degree, 0)
    vpxf    = savgol_filter(x,     win, pol_degree, 1) / edt
    vpyf    = savgol_filter(y,     win, pol_degree, 1) / edt
    vpfmag  = np.sqrt(vpxf**2 + vpyf**2)
    wf      = savgol_filter(theta, win, pol_degree, 1) / edt
    Thetaf  = savgol_filter(theta, win, pol_degree, 0)
    etaf    = savgol_filter(wf/vpfmag, win, pol_degree) * 1.45/100

    rx = Xf - vpyf / wf
    ry = Yf + vpxf / wf
    Rorbit = np.sqrt((vpyf / wf)**2+(vpxf / wf)**2)

    data = [time, Xf, Yf, Thetaf, wf, rx, ry, etaf, Rorbit]
    return edt, data

# ---------------------------- optimizer ----------------------------
def optimize_ls(l1, edt, data, erf_id, sim_time,
                fw_interp, fw, theta_interp, optype='residual'):
    df_opt = []
    vs     = []

    time, Xf, Yf, Thetaf, wf, rx, ry, etaf, Rorbit = data
    l1   = min(int(l1), len(time) - 1)
    dirr = np.sign(np.average(wf))

    error_compute = lambda params: error_function_general(
        params, [0, l1], erf_id, data, edt, theta_interp, fw, dirr, optype)

    if optype == 'residual':
        result = least_squares(error_compute, 100*np.random.rand(erf_param_len[erf_id]))
    elif optype == 'euclidian':
        result = minimize(error_compute, np.random.rand(erf_param_len[erf_id]),
                          method='L-BFGS-B')

    optimized_params = result.x
    fv1, fv2 = erfs[erf_id](optimized_params, theta_interp, fw_interp, dirr)
    print(result.x,
          np.round(result.fun if optype == 'euclidian' else result.cost, 5))
    vs.append(result.x)

    l0 = 0
    b0 = BBot(r=[Xf[l0], Yf[l0]], theta_rad=Thetaf[l0], t0=time[l0],
              vfunc1=fv1, vfunc2=fv2, wfunc=fw)
    df_sim = simulate_bbot(b0, sim_time, edt, 'CN', 'time')

    df_opt.append(df_sim)
    print(len(df_sim))
    return data, df_sim, result, vs

# ---------------------------- load + filter ------------------------
tr1   = np.load('../bbotv2_data/mode_1.npy').T
tr1   = tr1[120:, :]
tr1   = tr1 - tr1[0, :]
time, x, y, theta = tr1.T
x    *= 1/100
y    *= 1/100
theta = np.unwrap(theta, period=np.pi) + np.pi
N     = len(time)

edt, data = filter_data(time, x, y, theta)
time, Xf, Yf, Thetaf, wf, rx, ry, etaf, Rorbit = data

# ---------------------------- sweep --------------------------------
Pct_train = [1.0, 0.8, 0.6, 0.4]
Ntrain    = np.int32(np.array(Pct_train) * N)
Nmodes    = [10, 25, 50, 100]

np.random.seed(0)
results = {}
for i, k in enumerate(Nmodes):
    for j, (frac, Ntr) in enumerate(zip(Pct_train, Ntrain)):
        print(f'[{i},{j}] modes={k:3d}  Ntrain={Ntr} ({int(frac*100)}%)')

        w_interp, theta_interp, info = make_phi(
            wf, time, n_modes=k, frac=frac,
            anchor_t=time[0], anchor_value=Thetaf[0])

        edata_c, opt, res, vs = optimize_ls(
            Ntr, edt, data, erf_id=11,
            sim_time     = time[-1] - time[0],
            fw_interp    = w_interp,
            fw           = w_interp,
            theta_interp = theta_interp,
            optype       = 'residual')

        # store only picklable pieces — drop callables, keep DataFrames/arrays
        results[(i, j)] = dict(
            n_modes      = k,
            frac         = frac,
            Ntrain       = int(Ntr),
            opt          = opt,             # simulator output DataFrame
            params       = res.x,
            cost         = float(res.cost),
            vs           = vs,
            info         = info,
        )

# ---------------------------- save ---------------------------------
out_dir = '../code_outputs/data'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'sweep_modes_vs_ntrain.pkl')

payload = dict(
    Pct_train = Pct_train,
    Ntrain    = Ntrain.tolist(),
    Nmodes    = Nmodes,
    N         = N,
    time      = time,
    Xf        = Xf,
    Yf        = Yf,
    Thetaf    = Thetaf,
    wf        = wf,
    etaf      = etaf,
    edt       = edt,
    results   = results,
)
with open(out_path, 'wb') as f:
    pickle.dump(payload, f)
print(f'\nsaved → {out_path}')