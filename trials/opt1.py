# Single-shot version: one (n_modes, Ntrain) configuration, plotted directly.
import sys, os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.signal import savgol_filter

from src.Ellipse2dObj    import *
from src.animate         import *
from src.trajectory      import *
from src.simopt          import *
from src.optimizer_utils import *

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

# ---------------------------- KNOBS --------------------------------
K_MODES   = 100         # number of Fourier modes kept
PCT_TRAIN = 1.0        # fraction of trajectory used for optimization + interpolator
ERF_ID    = 11
SEED      = 0
SCF       = 100        # m -> cm for display

# ---------------------------- helpers ------------------------------
def make_phi(wf, time, n_modes, frac=1.0, anchor_t=None, anchor_value=None):
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
    if N_sub % 2 == 0: scales[-1] = 1.0

    ac_freq   = omega[1:]
    ac_coef_w = scales[1:] * W_filt[1:] / N_sub
    ac_coef_p = ac_coef_w / (1j * ac_freq)
    dc_value  = float(W_filt[0].real / N_sub)

    def wf_fn(t):
        t_arr = np.asarray(t, dtype=float); was_scalar = (t_arr.ndim == 0)
        tau   = np.atleast_1d(t_arr).ravel() - t0_ref
        ac    = (ac_coef_w[:, None]
                 * np.exp(1j*ac_freq[:, None]*tau[None, :])).sum(axis=0).real
        out   = dc_value + ac
        return float(out[0]) if was_scalar else out.reshape(t_arr.shape)

    def F(t):
        t_arr = np.asarray(t, dtype=float); was_scalar = (t_arr.ndim == 0)
        tau   = np.atleast_1d(t_arr).ravel() - t0_ref
        ac    = (ac_coef_p[:, None]
                 * np.exp(1j*ac_freq[:, None]*tau[None, :])).sum(axis=0).real
        out   = dc_value*tau + ac
        return float(out[0]) if was_scalar else out.reshape(t_arr.shape)

    F_anchor = F(anchor_t)
    def phi_fn(t): return F(t) - F_anchor + anchor_value

    info = dict(n_sub=N_sub, frac_used=N_sub/len(wf),
                t_fit_end=t_sub[-1], n_modes_kept=int(mask.sum()),
                period_sub=N_sub*dt_sub)
    return wf_fn, phi_fn, info


def error_function_general(sim_params, id, erf_id, data, dt,
                           theta_interp, w_interp, direction, optype='residual'):
    time, X, Y, T, wf, Rx, Ry, etaf, Rorbit = data
    x0, y0, phi_0 = X[id[0]], Y[id[0]], T[id[0]]
    t_0, t_f      = time[id[0]], time[id[1]]
    sim_time      = t_f - t_0

    fv1, fv2 = erfs[erf_id](sim_params, theta_interp, w_interp, direction)
    b0 = BBot(r=np.array([x0, y0]), theta_rad=phi_0, t0=t_0,
              vfunc1=fv1, vfunc2=fv2, wfunc=w_interp)
    df_sim = simulate_bbot(b0, sim_time, dt, 'CN', 'time')

    res_x = X[id[0]:id[1]] - df_sim.X
    res_y = Y[id[0]:id[1]] - df_sim.Y
    res_rx = Rx[id[0]:id[1]]     - df_sim.Gicrx
    res_ry = Ry[id[0]:id[1]]     - df_sim.Gicry
    res_rb = Rorbit[id[0]:id[1]] - np.sqrt(df_sim.vpmag/np.abs(df_sim.w))

    if optype == 'residual':
        return np.concatenate([res_x, res_y, res_rx, res_ry])
    return np.sum(np.sqrt(res_rx**2 + res_ry**2 + res_rb))


def filter_data(time, x, y, theta):
    win, pol_degree = 21, 3
    edt = np.average(np.diff(time))

    Xf     = savgol_filter(x,     win, pol_degree, 0)
    Yf     = savgol_filter(y,     win, pol_degree, 0)
    vpxf   = savgol_filter(x,     win, pol_degree, 1) / edt
    vpyf   = savgol_filter(y,     win, pol_degree, 1) / edt
    vpfmag = np.sqrt(vpxf**2 + vpyf**2)
    wf     = savgol_filter(theta, win, pol_degree, 1) / edt
    Thetaf = savgol_filter(theta, win, pol_degree, 0)
    etaf   = savgol_filter(wf/vpfmag, win, pol_degree) * 1.45/100

    rx     = Xf - vpyf / wf
    ry     = Yf + vpxf / wf
    Rorbit = np.sqrt((vpyf/wf)**2 + (vpxf/wf)**2)
    return edt, [time, Xf, Yf, Thetaf, wf, rx, ry, etaf, Rorbit]


def optimize_ls(l1, edt, data, erf_id, sim_time,
                fw_interp, fw, theta_interp, optype='residual'):
    time, Xf, Yf, Thetaf, wf, *_ = data
    l1   = min(int(l1), len(time) - 1)
    dirr = np.sign(np.average(wf))

    error_compute = lambda p: error_function_general(
        p, [0, l1], erf_id, data, edt, theta_interp, fw, dirr, optype)

    if optype == 'residual':
        result = least_squares(error_compute,
                               100*np.random.rand(erf_param_len[erf_id]))
    else:
        result = minimize(error_compute,
                          np.random.rand(erf_param_len[erf_id]),
                          method='L-BFGS-B')

    fv1, fv2 = erfs[erf_id](result.x, theta_interp, fw_interp, dirr)
    b0 = BBot(r=[Xf[0], Yf[0]], theta_rad=Thetaf[0], t0=time[0],
              vfunc1=fv1, vfunc2=fv2, wfunc=fw)
    df_sim = simulate_bbot(b0, sim_time, edt, 'CN', 'time')

    print(f'  params: {result.x}')
    print(f'  cost  : {result.cost:.5g}')
    return data, df_sim, result

# ---------------------------- load + filter ------------------------
tr1 = np.load('../bbotv2_data/mode_1.npy').T
tr1 = tr1[120:, :]
tr1 = tr1 - tr1[0, :]
time, x, y, theta = tr1.T
x    *= 1/100
y    *= 1/100
theta = np.unwrap(theta, period=np.pi) + np.pi
N     = len(time)

edt, data = filter_data(time, x, y, theta)
time, Xf, Yf, Thetaf, wf, rx, ry, etaf, Rorbit = data

# ---------------------------- one fit ------------------------------
Ntr = int(PCT_TRAIN * N)
np.random.seed(SEED)

print(f'modes={K_MODES}  Ntrain={Ntr} ({int(PCT_TRAIN*100)}% of {N})')
w_interp, theta_interp, info = make_phi(
    wf, time, n_modes=K_MODES, frac=PCT_TRAIN,
    anchor_t=time[0], anchor_value=Thetaf[0])
print(f'  make_phi: {info}')

_, opt, res = optimize_ls(
    Ntr, edt, data, erf_id=ERF_ID,
    sim_time     = (time[-1] - time[0])*2,
    fw_interp    = w_interp,
    fw           = w_interp,
    theta_interp = theta_interp,
    optype       = 'residual')

# ---------------------------- plot ---------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# top-left: xy trajectory
ax = axs[0, 0]
ax.plot(Xf*SCF,            Yf*SCF,            'k-',  lw=1.0, alpha=0.6,
        label='experiment')
ax.plot(opt.X[:Ntr]*SCF,   opt.Y[:Ntr]*SCF,   'C3-', lw=1.1,
        label='training window')
ax.plot(opt.X[Ntr:]*SCF,   opt.Y[Ntr:]*SCF,   'C4-', lw=1.1,
        label='predicted')
ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
ax.set_title(f'{K_MODES} modes, $N_{{train}} = {int(PCT_TRAIN*100)}\\%$',
             fontsize=11)

# top-right: x(t)
ax = axs[0, 1]
ax.plot(time,           Xf*SCF,          'k-',  lw=1.0, alpha=0.6)
ax.plot(opt.time[:Ntr], opt.X[:Ntr]*SCF, 'C3-', lw=1.1)
ax.plot(opt.time[Ntr:], opt.X[Ntr:]*SCF, 'C4-', lw=1.1)
ax.axvline(time[Ntr-1], color='gray', ls=':', lw=0.8)
ax.set_xlabel('t (s)'); ax.set_ylabel('x (cm)')
ax.grid(True, alpha=0.3)

# bottom-left: y(t)
ax = axs[1, 0]
ax.plot(time,           Yf*SCF,          'k-',  lw=1.0, alpha=0.6)
ax.plot(opt.time[:Ntr], opt.Y[:Ntr]*SCF, 'C3-', lw=1.1)
ax.plot(opt.time[Ntr:], opt.Y[Ntr:]*SCF, 'C4-', lw=1.1)
ax.axvline(time[Ntr-1], color='gray', ls=':', lw=0.8)
ax.set_xlabel('t (s)'); ax.set_ylabel('y (cm)')
ax.grid(True, alpha=0.3)

# bottom-right: angular velocity (measured vs Fourier interpolant)
ax = axs[1, 1]
ax.plot(time,           wf,                  'k-',  lw=1.0, alpha=0.6,
        label='measured')
ax.plot(time,           w_interp(time),      'C0-', lw=1.0,
        label='Fourier extension')
ax.axvline(time[Ntr-1], color='gray', ls=':', lw=0.8)
ax.set_xlabel('t (s)'); ax.set_ylabel(r'$\omega_f$ (rad/s)')
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.show()