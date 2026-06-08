import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import count
import random

print(os.getcwd())
#user defined
from src.Ellipse2dObj import *
from src.animate import *
from src.trajectory import *
from src.simopt import *

#constants
A1 = 5.5*0.5/100
A2 = 3.0*0.5/100
rho1 = 0.374
rho2 = 0.661
rmag = np.sqrt((rho1*A1)**2+(rho2*A2)**2)
Xmax = 53.4/100
Ymax = 48.4/100

# Global font settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "font.style": "italic" # Changed from italic to match standard thesis styles
})




def make_phi(wf, time, n_modes, frac=1.0, anchor_t=None, anchor_value=None):
    """
    Fit on the first `frac` of wf with top-n_modes Fourier bins.
    Returns (wf_fn, phi_fn, info), both callables on scalar or array t.
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
    
    freqs = np.fft.rfftfreq(N_sub, d=dt_sub)
    omega = 2*np.pi * freqs
    scales = np.full_like(freqs, 2.0)
    scales[0] = 1.0
    if N_sub % 2 == 0:
        scales[-1] = 1.0
    
    # AC mode constants (k >= 1); divide by i*omega to get the antiderivative
    ac_freq   = omega[1:]
    ac_coef_w = scales[1:] * W_filt[1:] / N_sub
    ac_coef_p = ac_coef_w / (1j * ac_freq)
    dc_value  = W_filt[0].real / N_sub   # wf mean → linear ramp slope for phi
    
    def wf_fn(t):
        t   = np.asarray(t, dtype=float)
        tau = t - t0_ref
        ac  = (ac_coef_w[:, None]
               * np.exp(1j * ac_freq[:, None] * tau.ravel()[None, :])
              ).sum(axis=0).real
        return (dc_value + ac).reshape(t.shape)
    
    def F(t):
        t   = np.asarray(t, dtype=float)
        tau = t - t0_ref
        ac  = (ac_coef_p[:, None]
               * np.exp(1j * ac_freq[:, None] * tau.ravel()[None, :])
              ).sum(axis=0).real
        return (dc_value * tau.ravel() + ac).reshape(t.shape)
    
    F_anchor = F(anchor_t)
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



# tr1 = np.load('../bbotv2_data/mode_1.npy').T
# tr1 = tr1[130:,:]
# tr1 = tr1-tr1[0,:]
# N  = len(tr1[:,0])


# phi = tr1[:, 3]
# time = tr1[:, 0]

# dt = np.mean(np.diff(time))

# phi = np.unwrap(tr1[:,3], period=np.pi)+np.pi  # in degrees, period 180
# wf = savgol_filter(phi, 21, 3, deriv=1)/dt
# wf_all = wf.copy()


# fractions = [0.40, 0.60, 0.80, 1.00]
# mode_grid = [10, 25, 50, 100]

# # build all reconstructions once (cheap; reuse for both grids)
# results = {}
# for frac in fractions:
#     for k in mode_grid:
#         wf_fn, phi_fn, info = make_phi(wf, time, n_modes=k, frac=frac,
#                                        anchor_t=time[0], anchor_value=phi[0])
#         results[(frac, k)] = (wf_fn(time), phi_fn(time), info)

# def plot_grid(signal, recon_key_idx, ylabel_sig, suptitle):
#     fig, axes = plt.subplots(len(fractions), len(mode_grid),
#                              figsize=(14, 9), sharex=True, sharey=True)
#     for i, frac in enumerate(fractions):
#         for j, k in enumerate(mode_grid):
#             ax   = axes[i, j]
#             rec  = results[(frac, k)][recon_key_idx]
#             info = results[(frac, k)][2]
#             ax.plot(time, signal, 'k-',  lw=0.8, alpha=0.4)
#             ax.plot(time, rec,    'C0-', lw=1.1)
#             if frac < 1.0:
#                 ax.axvline(info['t_fit_end'], color='C3',
#                            ls=':', lw=2, alpha=0.8)
#             ax.grid(True, alpha=0.3)
#             if i == 0:
#                 ax.set_title(f'{k} modes', fontsize=11)
#             if j == 0:
#                 ax.set_ylabel(f'fit on first {int(frac*100)}%\n' + ylabel_sig,
#                               fontsize=10)
#             if i == len(fractions) - 1:
#                 ax.set_xlabel('time [s]')
#     fig.suptitle(suptitle, fontsize=13)
#     plt.tight_layout()
#     plt.show()

# # # --- angular velocity grid (recon at index 0) ---
# plot_grid(wf,  0, r'$\omega_f$ [rad/s]', r'Angular velocity reconstruction')

# # # --- integrated angle grid (recon at index 1) ---
# plot_grid(phi, 1, r'$\phi$ [rad]',       r'Angle reconstruction (time-integrated)')