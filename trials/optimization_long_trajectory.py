
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
from src.optimizer_utils import *

#load data
from src import bbotData
#from bbotData import Ellipsoid
dfs, dtime, mps, la = bbotData.load()
sys.path.append(os.path.abspath('..'))
from scipy.interpolate import interp1d
from scipy.stats import norm, skewnorm

#constants
A1 = 5.5*0.5/100
A2 = 3.0*0.5/100
rho1 = 0.374
rho2 = 0.661
rmag = np.sqrt((rho1*A1)**2+(rho2*A2)**2)
Xmax = 53.4/100
Ymax = 48.4/100
from scipy.optimize import least_squares, minimize

DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "font.style": "italic" # Changed from italic to match standard thesis styles
})



def error_function_general(sim_params, id, erf_id,
                           data, dt, theta_interpolator, w_interpolator, 
                           direction, optype='residual', sim_order=1, fw_type='interpolate'):
    time, X,Y,T, wf,Rx,Ry, etaf = data
    x0, y0, phi_0 = X[id[0]], Y[id[0]], T[id[0]]
    t_0 = time[id[0]]
    t_f = time[id[1]]
    sim_time = t_f-t_0
    fv1, fv2 = erfs[erf_id](sim_params, theta_interpolator,w_interpolator, direction)
    if sim_order==1:
        b0 = Ell2D(r = np.array([x0,y0]), theta_rad = phi_0, t0=t_0, vfunc1=fv1, vfunc2=fv2, wfunc=w_interpolator)
        df_sim = b0.simulate(sim_time, dt, 'leg')
    elif sim_order==2:
        if fw_type == 'interpolate':
            b0 = BBot(r = np.array([x0,y0]), theta_rad = phi_0, t0=t_0, vfunc1=fv1, vfunc2=fv2, wfunc=w_interpolator)
            df_sim = simulate_bbot(b0, sim_time, dt, 'CN', 'time')
        elif fw_type == 'fft':
            b0 = BBot(r = np.array([x0,y0]), theta_rad = phi_0, t0=t_0, vfunc1=fv1, vfunc2=fv2, wfunc=w_interpolator)
            df_sim = simulate_bbot(b0, sim_time, dt, 'CN', 'angle')

    residual_x=(X[id[0]:id[1]] - df_sim.X)
    residual_y=(Y[id[0]:id[1]] - df_sim.Y)
    
    if optype == 'residual':
        return np.concatenate([residual_x, residual_y,])  # 
    elif optype == 'euclidian':
        return np.sum(np.sqrt(residual_x**2+residual_y**2))



def optimize_ls(l1, time, x, y, theta, erf_id, sim_time, normalize=False, optype='residual', fw_type = 'interpolate', sim_order=1):
    edt = np.average(np.diff(time))
    # time = t
    
    win=21
    pol_degree=3
    
    Xf = savgol_filter(x, win,pol_degree, 0)
    Yf = savgol_filter(y, win,pol_degree, 0)
    vpxf = savgol_filter(x, win,pol_degree, 1)/edt
    vpyf = savgol_filter(y, win,pol_degree, 1)/edt
    vpfmag = np.sqrt(vpxf**2+vpyf**2)
    wf = savgol_filter(theta, win,pol_degree, 1)/edt
    Thetaf = savgol_filter(theta, win,pol_degree, 0)
    etaf = savgol_filter(wf/vpfmag, win,pol_degree)*1.45/100#wf*1.45/100/np.sqrt(vpxf**2+vpyf**2)#savgol_filter(df.w/df.vmag, 21,3)*1.45/100#
    
    dirr = np.sign(np.average((wf)))

    rx = Xf - vpyf / wf
    ry = Yf + vpxf / wf
    # data = [time[:l1], Xf[:l1], Yf[:l1], Thetaf[:l1], wf[:l1], rx[:l1], ry[:l1], etaf[:l1]]
    # keep arrays full
    data = [time, Xf, Yf, Thetaf, wf, rx, ry, etaf]
    # clamp l1 so time[l1] is valid (the 1.0*len(x) case is otherwise OOB)
    l1 = min(int(l1), len(time) - 1)
    
    df_opt = []

    vs = []
    # l0 = 0
    # l1 = len(x) - 1
    # sim_time = time[l1] - time[l0]

    # Set up interpolators for angular variables
    if fw_type=='interpolate':
        fw_interp = interp1d(time, wf, kind='cubic', fill_value="extrapolate")
        theta_interp = interp1d(time, Thetaf, kind='cubic', fill_value="extrapolate")
        fw = lambda t: fw_interp(t).item()
        theta_func = lambda t: theta_interp(t).item()
    elif fw_type=='fft':
        print('none')
        

    # Error function closure
    error_compute = lambda params: error_function_general(
        params, [0, l1], erf_id, data, edt, theta_interp, fw, dirr, optype, sim_order, fw_type
    )

    # Choose optimizer based on optype
    if optype == 'residual':
        result = least_squares(error_compute, 100*np.random.rand(erf_param_len[erf_id]))
    elif optype == 'euclidian':
        result = minimize(error_compute, np.random.rand(erf_param_len[erf_id]), method='L-BFGS-B')

    optimized_params = result.x
    fv1, fv2 = erfs[erf_id](optimized_params, theta_interp, fw_interp, dirr)
    print(result.x, np.round(result.fun if optype == 'euclidian' else result.cost, 5))

    vs.append(result.x)

    l0=0
    if sim_order==1:
        b0 = Ell2D(r=[Xf[l0], Yf[l0]], theta_rad=Thetaf[l0], t0=time[l0],
                     vfunc1=fv1, vfunc2=fv2, wfunc=fw)
        df_sim = b0.simulate(sim_time, edt, 'leg')
    elif sim_order==2: 
        if fw_type=='interpolate':
            b0 = BBot(r=[Xf[l0], Yf[l0]], theta_rad=Thetaf[l0], t0=time[l0],
                        vfunc1=fv1, vfunc2=fv2, wfunc=fw)
            
            df_sim  = simulate_bbot(b0, sim_time, edt, 'CN', 'time')
        if fw_type=='fft':
            b0 = BBot(r=[Xf[l0], Yf[l0]], theta_rad=Thetaf[l0], t0=time[l0],
                        vfunc1=fv1, vfunc2=fv2, wfunc=fw)
            
            df_sim  = simulate_bbot(b0, sim_time, edt, 'CN', 'angle')
            b0.df=df_sim
            b0.postprocess()
    df_opt.append(df_sim)
    print(len(df_sim))
    return data, df_sim, result, vs


tr1 = np.load('../bbotv2_data/mode_1.npy').T
tr1 = tr1[120:,:]
tr1 = tr1-tr1[0,:]

time, x, y, theta = tr1[:,:].T
x *= 1/100
y *= 1/100
theta = np.unwrap(theta, period=np.pi)+np.pi   # in degrees, period 1

print('data len', len(x))

# edata,  simdf, res, vs = optimize_ls(time, x, y, theta, 11, 'residual', fw_type='interpolate', sim_order=2)




edatas = []
opts   = []
ress   = []
errors = []
vparams = []
rows = 0
ijk = [ [2,0,8],[4,1,5], [0,0,5],[2,1,7]]
models = [11,11,11,11]
ls = np.int32(np.array([1.0, 0.8, 0.6, 0.4])*len(x))

for c in range(4):    
    rows+=1
    edata, opt8, res, vs = optimize_ls(ls[c], time, x, y, theta, 11, time[-1]-time[0], 'residual', fw_type='interpolate', sim_order=2)
    edatas.append(edata)
    opts.append(opt8)
    ress.append(res)
    vparams.append(vs)
    l1  = len(edata[1])-1

# print(len(opt8.X))
# plt.plot(opt8.X, opt8.Y)
# plt.figure()
# plt.plot(opt8.X, opt8.Y)

# # plt.plot(opt8.X[ls[c]:],opt8.Y[ls[c]:] , ls='--', alpha=0.5)

# plt.show()


errors = np.array(errors)
fig, axs = plt.subplots(2, 4, figsize=(12, 5))  # 2 rows, 5 columns
fig.subplots_adjust(hspace=0.4, wspace=0.3)  # vertical spacing

cols = 4
errors = []
params = []
letters = 'abcdefghijkl'
shift=0
shift_delta = lambda x: 1.01*(np.max(x)-np.min(x))
cntr=0

scf = 100
# for c in range(4):

#     edata = edatas[c]
#     opt8 = opts[c]
#     res = ress[c]
#     vs = vparams[c]

#     xrange = [x.min() * scf, x.max() * scf]
#     yrange = [y.min() * scf, y.max() * scf]
#     delta = max(xrange[1] - xrange[0], yrange[1] - yrange[0])
#     xrange[1] = xrange[0] + delta
#     yrange[1] = yrange[0] + delta

#     ax_top = axs[0, c]
#     ax_bottom = axs[1, c]
    
#     ax_top.plot(x*scf, y*scf, c='black', lw=1)
#     # ax_top.plot(edata[1]*scf, edata[2]*scf, c='black', lw=1)
#     ax_top.plot((opt8.X[:ls[c]] + shift_delta(opt8.X[:ls[c]]) * shift)*scf, opt8.Y[:ls[c]]*scf, c='C3', lw=1)
#     ax_top.plot((opt8.X[ls[c]:] + shift_delta(opt8.X[ls[c]:]) * shift)*scf, opt8.Y[ls[c]:]*scf, c='C4', lw=1)
#     # ax_top.plot((opt8.X[:] + shift_delta(opt8.X[:]) * shift)*scf, opt8.Y[:]*scf, c='C3')

#     ax_top.set_xlabel('x (cm)')
#     if c == 0:
#         ax_top.set_ylabel('y (cm)', )
#     ax_top.axis('equal')
#     ax_top.text(0.85, 0.9, f'({letters[c]})', transform=ax_top.transAxes)

#     ax_bottom.plot(edata[0], np.abs(edata[7]), c='black', lw=1)
#     ax_bottom.plot(opt8.time[:ls[c]], opt8.sigma[:ls[c]], c='C3')
#     ax_bottom.plot(opt8.time[ls[c]:], opt8.sigma[ls[c]:], c='C4')

#     ax_bottom.set_xlabel('t (s)')
#     if c == 0:
#         ax_bottom.set_ylabel(r'$\eta$')
#     ax_bottom.text(0.866, 0.9, f'({letters[c+4]})', transform=ax_bottom.transAxes)
#     # ax_bottom.set_ylim([-0.1, 2])

# fig.savefig(f'../code_outputs/figures/longdatafit.pdf', dpi=600)
# plt.show()


errors = np.array(errors)
fig, axs = plt.subplots(4, 4, figsize=(10, 10))  # 2 rows, 5 columns
fig.subplots_adjust(hspace=0.4, wspace=0.3)  # vertical spacing

for c in range(4):

    edata = edatas[c]
    opt8 = opts[c]
    res = ress[c]
    vs = vparams[c]

    xrange = [x.min() * scf, x.max() * scf]
    yrange = [y.min() * scf, y.max() * scf]
    delta = max(xrange[1] - xrange[0], yrange[1] - yrange[0])
    xrange[1] = xrange[0] + delta
    yrange[1] = yrange[0] + delta

    ax_top = axs[0, c]
    ax_bottom = axs[1, c]
    ax3 = axs[2, c]
    ax4 = axs[3, c]

    ax_top.plot(x*scf, y*scf, c='black', lw=1)
    # ax_top.plot(edata[1]*scf, edata[2]*scf, c='black', lw=1)
    ax_top.plot((opt8.X[:ls[c]] + shift_delta(opt8.X[:ls[c]]) * shift)*scf, opt8.Y[:ls[c]]*scf, c='C3', lw=1)
    ax_top.plot((opt8.X[ls[c]:] + shift_delta(opt8.X[ls[c]:]) * shift)*scf, opt8.Y[ls[c]:]*scf, c='C4', lw=1)
    # ax_top.plot((opt8.X[:] + shift_delta(opt8.X[:]) * shift)*scf, opt8.Y[:]*scf, c='C3')

    ax_top.set_xlabel('x (cm)')
    if c == 0:
        ax_top.set_ylabel('y (cm)', )
    ax_top.axis('equal')
    ax_top.text(0.85, 0.9, f'({letters[c]})', transform=ax_top.transAxes)

    ax_bottom.plot(edata[0], np.abs(edata[7]), c='black', lw=1)
    ax_bottom.plot(opt8.time[:ls[c]], opt8.sigma[:ls[c]], c='C3')
    ax_bottom.plot(opt8.time[ls[c]:], opt8.sigma[ls[c]:], c='C4')

    ax_bottom.set_xlabel('t (s)')
    if c == 0:
        ax_bottom.set_ylabel(r'$\eta$')
    ax_bottom.text(0.866, 0.9, f'({letters[c+4]})', transform=ax_bottom.transAxes)
    # ax_bottom.set_ylim([-0.1, 2])
    
    ax3.plot(time, x*scf, c='black', lw=1)
    ax3.plot(opt8.time[:ls[c]], opt8.X[:ls[c]]*scf, c='C3', lw=1)
    ax3.plot(opt8.time[ls[c]:], opt8.X[ls[c]:]*scf, c='C4', lw=1)
    ax3.set_xlabel('t (s)')
    ax3.set_ylabel('x (cm)')

    ax4.plot(time, y*scf, c='black', lw=1)
    ax4.plot(opt8.time[:ls[c]], opt8.Y[:ls[c]]*scf, c='C3', lw=1)
    ax4.plot(opt8.time[ls[c]:], opt8.Y[ls[c]:]*scf, c='C4', lw=1)
    ax4.set_xlabel('t (s)')
    ax4.set_ylabel('y (cm)')


fig.savefig(f'../code_outputs/figures/longdatafit_xy.pdf', dpi=600)
plt.show()