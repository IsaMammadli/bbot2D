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


tr1 = np.load('../bbotv2_data/mode_1.npy').T
tr1 = tr1[120:,:]
tr1 = tr1-tr1[0,:]
# tr2 = np.load('../bbotv2_data/mode_2.npy').T


phi = tr1[:, 3]
time = tr1[:, 0]



phi = np.unwrap(tr1[:,3], period=np.pi)   # in degrees, period 180
wf = savgol_filter(phi, 31, 3, deriv=1)/np.gradient(time)
# plt.plot(np.gradient(phi)/np.gradient(time))
# plt.plot(wf)
# print(np.average(wf), np.average(np.gradient(time)), np.max(time))

r0_init = np.array([tr1[0, 1],  tr1[0, 2]])
phi_0_init = phi[0]+np.pi
T = 70
dt = 0.01
def run_simulation(fw, fv1, fv2):
    #initial position and orientation
    r0 = r0_init.copy()
    phi_0 = phi_0_init
    b =  BBot(r = r0, theta_rad = phi_0, t0=0, vfunc1=fv1, vfunc2=fv2, wfunc=fw)
    df = simulate_bbot(b, T, dt, 'CN', 'time', False)
    return df

# simulation parameters
fw  = lambda t: -3.264 #experimental average angular velocity
nu = -0.06 #determines the radius of larger loop
alpha0 = np.pi #direction
fv1 = lambda t:  0.03+0.002*np.cos(fw(t)*t+nu*t+alpha0)
fv2 = lambda t:  0.03-0.002*np.sin(fw(t)*t+nu*t+alpha0)
df0 = run_simulation(fw, fv1, fv2)



#plotting
plt.plot(tr1[:, 1], tr1[:, 2], c='k')
plt.plot(df0.X*100, df0.Y*100, c='C3')
print(tr1[0, :])
print(df0.X[0], df0.Y[0])

plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.savefig('../code_outputs/figures/badtrajectoryfit.pdf', dpi=300)
plt.show()

