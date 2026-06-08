# plot_sweep.py
import os, pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size":   12,
    "font.style":  "italic",
})

in_path = '../code_outputs/data/sweep_modes_vs_ntrain.pkl'
with open(in_path, 'rb') as f:
    P = pickle.load(f)

Pct_train = P['Pct_train']
Nmodes    = P['Nmodes']
time      = P['time']
Xf        = P['Xf']
Yf        = P['Yf']
results   = P['results']
scf       = 100   # m -> cm

# ---------------------------- xy grid ------------------------------
# xy grid — DROP sharex/sharey
fig, axs = plt.subplots(len(Nmodes), len(Pct_train),
                        figsize=(12, 12))
fig.subplots_adjust(hspace=0.25, wspace=0.15)

for i, k in enumerate(Nmodes):
    for j, frac in enumerate(Pct_train):
        ax  = axs[i, j]
        R   = results[(i, j)]
        opt = R['opt']
        Ntr = R['Ntrain']

        ax.plot(Xf*scf,           Yf*scf,           'k',  lw=1.0,
                label='experiment')
        ax.plot(opt.X[:Ntr]*scf,  opt.Y[:Ntr]*scf,  'C3-', lw=1.1,
                label='training window')
        ax.plot(opt.X[Ntr:]*scf,  opt.Y[Ntr:]*scf,  'C4-', lw=1.1,
                label='predicted')

        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title(f'$N_{{train}} = {int(frac*100)}\\%$', fontsize=11)
        if j == 0:
            ax.set_ylabel(f'{k} modes\ny (cm)', fontsize=10)
        if i == len(Nmodes) - 1:
            ax.set_xlabel('x (cm)')

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.005))

out_dir = '../code_outputs/figures'
os.makedirs(out_dir, exist_ok=True)
plt.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(os.path.join(out_dir, 'sweep_xy.pdf'),
            dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------- x(t) grid ----------------------------
fig, axs = plt.subplots(len(Nmodes), len(Pct_train),
                        figsize=(14, 10), sharex=True, sharey=True)
for i, k in enumerate(Nmodes):
    for j, frac in enumerate(Pct_train):
        ax  = axs[i, j]
        R   = results[(i, j)]
        opt = R['opt']
        Ntr = R['Ntrain']

        ax.plot(time, Xf*scf, 'k-', lw=1.0)
        ax.plot(opt.time[:Ntr], opt.X[:Ntr]*scf, 'C3-', lw=1.1)
        ax.plot(opt.time[Ntr:], opt.X[Ntr:]*scf, 'C4-', lw=1.1)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f'$N_{{train}} = {int(frac*100)}\\%$', fontsize=11)
        if j == 0:
            ax.set_ylabel(f'{k} modes\nx (cm)', fontsize=10)
        if i == len(Nmodes) - 1:
            ax.set_xlabel('t (s)')

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'sweep_x_t.pdf'),
            dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------- y(t) grid ----------------------------
fig, axs = plt.subplots(len(Nmodes), len(Pct_train),
                        figsize=(14, 10), sharex=True, sharey=True)
for i, k in enumerate(Nmodes):
    for j, frac in enumerate(Pct_train):
        ax  = axs[i, j]
        R   = results[(i, j)]
        opt = R['opt']
        Ntr = R['Ntrain']

        ax.plot(time, Yf*scf, 'k-', lw=1.0)
        ax.plot(opt.time[:Ntr], opt.Y[:Ntr]*scf, 'C3-', lw=1.1)
        ax.plot(opt.time[Ntr:], opt.Y[Ntr:]*scf, 'C4-', lw=1.1)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f'$N_{{train}} = {int(frac*100)}\\%$', fontsize=11)
        if j == 0:
            ax.set_ylabel(f'{k} modes\ny (cm)', fontsize=10)
        if i == len(Nmodes) - 1:
            ax.set_xlabel('t (s)')

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'sweep_y_t.pdf'),
            dpi=600, bbox_inches='tight')
plt.show()

# ---------------------------- cost heatmap -------------------------
cost = np.array([[results[(i, j)]['cost']
                  for j in range(len(Pct_train))]
                 for i in range(len(Nmodes))])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(np.log10(cost), aspect='auto', cmap='viridis')
ax.set_xticks(range(len(Pct_train)))
ax.set_xticklabels([f'{int(p*100)}%' for p in Pct_train])
ax.set_yticks(range(len(Nmodes)))
ax.set_yticklabels(Nmodes)
ax.set_xlabel(r'$N_{train}$')
ax.set_ylabel('# modes')
ax.set_title(r'$\log_{10}$(least-squares cost)')
for i in range(len(Nmodes)):
    for j in range(len(Pct_train)):
        ax.text(j, i, f'{cost[i,j]:.2g}',
                ha='center', va='center',
                color='white' if np.log10(cost[i,j]) < np.log10(cost).mean()
                              else 'black',
                fontsize=9)
fig.colorbar(im, ax=ax)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'sweep_cost_heatmap.pdf'),
            dpi=600, bbox_inches='tight')
plt.show()



