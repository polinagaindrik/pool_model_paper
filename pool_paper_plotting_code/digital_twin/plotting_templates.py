#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from digital_twin.model import *

colors_all = {
        'R': '#808080',
        'N_A':'#D06062',
        'N_B': '#4E89B1',
        'N':'#7E57A5',
        'T':'#99582A',
        'T_A':'#c79758',
        'N_lambd_1e-2_omega_0':'#E2B100',
        'N_lambd_1e-3_omega_0':'#386641',
        'N_lambd_1e-3_omega_0_5':'#0982A4',
        'N_wo_tempshift':'#679E48',
        'N_tempshift_10':'#ED733E',
        'N_tempshift_10_5_15':'#C3568A',
        'N_Lm': '#ED733E',
        'N_Lm_woT':'#D70040',
        'N_Ls23K': '#679E48',
        'N_Ls23Kco': '#386641',
        'N_Lm_withT':'#D06062',
        'N_LsCTC494': '#4E89B1',
        'N_LsCTC494co': '#00356B',
    }


figsize_default = (6.5, 4.0)
figsize_default_small = (6.5, 2.0)
figsize_default2subpl = (13, 4.0)

plt.rcParams['figure.dpi'] = 400
plt.rcParams["font.family"] = "serif"
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

plt.rcParams['legend.fontsize'] = 15.
plt.rcParams['legend.framealpha'] = 0.
plt.rcParams['legend.handlelength'] = 1.8
plt.rcParams['axes.prop_cycle'] = plt.cycler(linewidth=[2.5])
plt.rcParams['font.size'] = 15
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)

def pool_paper_plotting_template(fig, ax):
    return fig, ax

def set_labels(fig, ax, xlabel, y_label):
    ax.set_xlabel(xlabel, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_ylabel(y_label, fontsize=15)
    return fig, ax

def plot_cost_function(df_optim, path='', add_name=''):
    fig, ax = plt.subplots()
    ax.plot(df_optim['iteration'], df_optim['cost'])
    ax.set_xlabel('optimization step', fontsize=12)
    ax.set_ylabel('cost function', fontsize=12)
    ax.set_yscale('log')
    plt.savefig(path+f'cost_plot{add_name}.png')#, bbox_inches='tight')
    plt.close(fig)

############## Plotting ######################333
def plot_all_curves(param_ode, x10, data=None, path='', add_name=''):
    clrs = [colors_all['N_A'], colors_all['N_B'], colors_all['R'], colors_all['T'], colors_all['N']]
    n_cl = 3
    if data is not None:
        days, [obs_x] = extract_observables_from_df([data])
    t = np.linspace(days[0], days[-1], 100)
    x0 = set_initial_vals(x10, None, n_cl, pH0=obs_x[0][-1][0])
    pH_series = np.array([obs_x[0][-1], days]).T
    x_sol = model_ODE_solution(ode_model_coculture, t, param_ode, x0, [pH_series, n_cl])
    obs_model = observable(days, x_sol)
    lbls = ["ls", "lm", "BAC", "LA", "pH"]
    fig, ax = plt.subplots()
    for i in range(2):
        ax.plot(t, obs_model[i], label=lbls[i], color=clrs[i])
        if data is not None:
            ax.scatter(days, obs_x[0][i], label=lbls[i]+'_data', marker='x', color=clrs[i])
    # ax.plot(t, x_sol[2], label='R')

    ax.plot(t, x_sol[4], label='R', linestyle='dashed', color=clrs[4])
    ax.plot(t, x_sol[2], label='lm_sen', linestyle='dotted', color=clrs[2])
    ax.plot(t, x_sol[3], label='lm_res', linestyle='dotted', color=clrs[3])
    ax.set_yscale("log")
    #ax.set_ylim(10**-3, 10**9)
    plt.legend()
    plt.savefig(path + f"x_sol_R{add_name}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, obs_model[2], label=lbls[2], color=clrs[2])
    if data is not None:
        ax.scatter(days, obs_x[0][2], label=lbls[2]+'_data', marker='x', color=clrs[2])
    plt.legend()
    plt.savefig(path + f"BAC{add_name}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, obs_model[3], label=lbls[3], color=clrs[3])
    if data is not None:
        ax.scatter(days, obs_x[0][3], label=lbls[3]+'_data', marker='x', color=clrs[3])
        ax.scatter(days, obs_x[0][4], label=lbls[4]+'_data', marker='x', color=clrs[4])
    plt.legend()
    plt.savefig(path + f"LA_pH{add_name}.png", bbox_inches="tight")
    plt.close(fig)

########### pH(LA) # function
def pH_LA_dependence(days, LA_data, pH_data, add_name='', path=''):
    pH0 = pH_data[0]
    K_a = 1.38*10**(-4)
    pH = pH0 + np.log(- K_a + np.sqrt(K_a**2 + 4 * K_a*LA_data)/2)
    #print(- K_a + np.sqrt(K_a**2 + 4 * K_a*LA_data))
    fig, ax = plt.subplots()
    ax.scatter(days, pH, label='pH(LA)')
    ax.scatter(days, pH_data, label='pH_data', marker='x')
    ax.scatter(days, LA_data, label='LA_data', marker='x')
    plt.legend()
    plt.savefig(path + f"LA_pH{add_name}.png", bbox_inches="tight")
    plt.close(fig)
    return pH