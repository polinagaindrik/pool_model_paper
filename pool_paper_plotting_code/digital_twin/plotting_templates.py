#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

# Set some custom colors/scatter markers for plotting (mb add more or remove later)
colors = [(0 / 255, 102 / 255, 217 / 255),
          (90 / 255, 170 / 255, 90 / 255),
          (215 / 255, 48 / 255, 39 / 255)]  
gray = (128 / 255, 128 / 255, 128 / 255)
gray2 = (160 / 255, 160 / 255, 160 / 255)
scatter_marker = ['^', 'o', 'D', "s", "X", ".", "1", "2", "3", "4", "*", "P", "d", "+", "x"]


def template_plot_measurements0(ax0, temp, data, c='b'):
    n_plot = 0
    for d in data: 
        if d['const'] == temp:
            for meas, std in zip(d['obs_mean'], d['obs_std']):
                ax0.errorbar(d['times'], np.log10(meas), fmt=scatter_marker[n_plot],  yerr=np.abs(0.43*std/meas),
                             markersize=7, color=c, label="T = {}".format(temp))
            n_plot += 1


def template_fig_for_many_temps(temp_plot, xlabel, ylabel, time_lim=[]):
    fig, ax = plt.subplots(1, len(temp_plot), figsize=(4.8*len(temp_plot), 5), sharey=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.15)
    for i, ax0 in enumerate(ax): 
        ax0.tick_params(axis='x', which='major', labelsize=13, labelrotation=0)
        if len(time_lim) == len(temp_plot):
            ax0.set_xlim(-0.2, time_lim[i])
        fig, ax0 = set_labels(fig, ax0, xlabel, ylabel)
    return fig, ax


def template_fig_1_temp(temp_plot, xlabel, ylabel, tcr, time_lim=[]):
    fig, ax = plt.subplots()
    if len(time_lim) != 0:
        ax.set_xlim(-0.2, time_lim[0])
        ticks_val = [int(4*j) for j in range (int(time_lim[0]*0.25)+1)]
        tick_label = [f'{round(day)}' for day in ticks_val]
        ax.set_xticks(ticks_val)
        ax.set_xticklabels(tick_label)
        ax.tick_params(axis='x', which='major', labelsize=13, labelrotation=0)
        if len(tcr) != 0:
            ax.set_xticks(tcr, minor = True)
            ax.set_xticklabels([f"{round(t)}" for t in tcr], minor=True)
            ax.tick_params(axis='x', which='minor', labelsize=15, labelcolor=colors[-1], labelrotation=0)
    fig, ax = set_labels(fig, ax, xlabel, ylabel)
    return fig, [ax]


def set_labels(fig, ax, xlabel, y_label):
    ax.set_xlabel(xlabel, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_ylabel(y_label, fontsize=15)
    return fig, ax


def template_plot_model(ax0, time, estim, low, upp, c='b', lab='', labub=''):
    for tr_est, tr_u, tr_b in zip(estim, upp, low):
        ax0.plot(time, np.log10(tr_est), linestyle='solid', color=c, label=lab)
        ax0.plot(time, np.log10(tr_u), linestyle='dashed', color=gray2, label=labub)
        ax0.plot(time, np.log10(tr_b), linestyle='dashed', color=gray)

    ax0.set_ylim(np.log10(np.min(low))*0.9, np.log10(np.max(upp))*1.2)