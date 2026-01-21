# fmt: off
#!/usr/bin/env python3
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/pool_paper_plotting_code")
from digital_twin import *


def pool_model_3pools(t, x, param, x0, const):
    (lambd, omega1, mu, omega2, ) = const
    (L0, G0, D0, ) = x0
    (L, G, D, ) = x

    return [
        - (omega1 + lambd) * L,
        lambd * L + (mu - omega2) * G,
        omega1 * L + omega2 * G
        ]

def pool_model_3pools_resource(t, x, param, x0, const):
    (lambd, omega1, mu, omega2, N_t, ) = const
    (L0, G0, D0, ) = x0
    (L, G, D, ) = x

    return [
        - (omega1 + lambd) * L,
        lambd * L + (mu - mu * ((L + G + D)/N_t) - omega2) * G,
        omega1 * L + omega2 * G
        ]

def pool_model_2pools_resource_lag(t, x, param, x0, const):
    (lambd, mu, N_t, ) = const
    (L0, G0, R0, ) = x0
    (L, G, R ) = x

    return [
        - lambd * R * L,
        lambd * R * L + mu * R * G,
        - (mu / N_t) * R * G
        ]

def pool_model_2pools_resource_lag2(t, x, param, x0, const):
    (lambd, mu, N_t, ) = const
    (L0, G0, ) = x0
    (L, G, ) = x
    r_term = 1 - (L+G)/N_t
    return [
        - lambd * r_term * L,
        lambd * r_term * L + mu * r_term * G,
        ]

def pool_model_dormant(t, x, param, x0, const):
    T_const = 12.
    (lambdT, omega1, muT, omega2, xi, N_t, sigma, delta, zeta, alpha, shift_cnd,) = const
    (L0, G0, D0, S0, ) = x0
    (L, G, D, S, ) = x
    if len(shift_cnd) != 0:
        gamma, Temp_change = temp_stress(t, sigma, delta, zeta, alpha, shift_cnd)
        lambd = lambdT * (T_const+Temp_change)
        mu = muT * (T_const+Temp_change)
    else:
        gamma = 0
        lambd = lambdT * T_const
        mu = muT * T_const
    return [
        - (omega1 + lambd) * L,
        lambd * L + (mu - mu * ((L + G + D)/N_t) - omega2 - gamma) * G + xi * S,
        omega1 * L + omega2 * G,
        gamma * G - xi * S
        ]

def pool_model_tempstress(t, x, param, x0, const):
    (lambd, omega1, mu, omega2, N_t, sigma, delta, zeta, alpha, shift_cnd,) = const
    (L0, G0, D0, ) = x0
    (L, G, D, ) = x

    gamma = temp_stress(t, sigma, delta, zeta, alpha, shift_cnd)
    return [
        - (omega1 + lambd) * L + gamma * G,
        lambd * L + (mu - mu * ((L + G + D)/N_t) - omega2 - gamma) * G,
        omega1 * L + omega2 * G
        ]

def temp_stress(t, sigma, delta, zeta, alpha, shift_cnd):
    #shift_cnd = ((t, T), (t, T))
    gamma = 0
    Temp_change = 0
    for tsh, Tsh in shift_cnd:
        if t >= tsh:
            if Tsh > 0:
                f = alpha*zeta*Tsh
            else:
                f = -(1-alpha)*zeta*Tsh
            gamma += sigma * f * np.exp(-delta * (t - tsh))
        if np.abs(t - tsh) <= 0.1:
            Temp_change = Tsh
    return gamma, Temp_change

def pool_model_resource_comp(t, x, param, x0, const):
    (lambd_A, mu_A, lambd_B, mu_B, N_t, ) = const
    (L_A0, G_A0, L_B0, G_B0, R0, ) = x0
    (L_A, G_A, L_B, G_B, R, ) = x

    return [
        - lambd_A * R * L_A,
          lambd_A * R * L_A + mu_A * R * G_A,
        - lambd_B * R * L_B,
          lambd_B * R * L_B + mu_B * R * G_B,
        - (mu_A / N_t) * R * G_A - (mu_B / N_t) * R * G_B
        ]

def pool_model_2sp_comp_toxin1(t, x, param, x0, const):
    (lambd_A, mu_A, lambd_B, mu_B, N_t, k, omega,) = const
    (L_A0, G_A0, L_B0, G_B0, R0, T0) = x0
    (L_A, G_A, L_B, G_B, R, T) = x

    return [
        - lambd_A * R * L_A,
          lambd_A * R * L_A + (mu_A * R - omega/(k*N_t) * T) * G_A,
        - lambd_B * R * L_B,
          lambd_B * R * L_B + (mu_B * R) * G_B,
        - (mu_A / N_t) * R * G_A - (mu_B / N_t) * R * G_B,
          k * G_B
        ]

def pool_model_2sp_comp_toxin2(t, x, param, x0, const):
    (lambd_A, mu_A, lambd_B, mu_B, N_t, k, omega,) = const
    (L_A0, G_A0, L_B0, G_B0, R0, T0) = x0
    (L_A, G_A, L_B, G_B, R, T) = x

    return [
        - lambd_A * R * L_A,
          lambd_A * R * L_A + (mu_A * R - omega/(k*N_t) * T) * G_A,
        - lambd_B * R * L_B,
          lambd_B * R * L_B + (mu_B * R) * G_B,
        - (mu_A / N_t) * R * G_A - (mu_B / N_t) * R * G_B,
          k * G_B  * R # * mu_B
        ]

def pool_model_2sp_comp_inhib(t, x, param, x0, const):
    (psi, ) = const
    (G_A, G_B, ) = x

    return [
        G_A * (1 - G_A - G_B) / G_B,
        G_B * psi * (1 - G_A - G_B) / G_A
        ]

def pool_model_2sp_cooper(t, x, param, x0, const):
    (psi, phi, ) = const
    (G_A, G_B, W_A, W_B, ) = x
    return [
        W_B * G_A * (1 - G_A - G_B),
        psi * W_A * G_B * (1 - G_A - G_B),
        G_A * (1 - G_A - G_B),
        phi * G_B * (1 - G_A - G_B)
        ]

def observable_3pool(output):
    # Output (n_variables x n_times)
    # Variables: (L, G, D,) = x
    # => Output [n = L+G]
    return [output[0]+output[1]]#+output[2]]

def observable_3pool_dormant(output):
    # Output (n_variables x n_times)
    # Variables: (L, G, D,) = x
    # => Output [n = L+G]
    return [output[0]+output[1]+output[3]]#]

def observable_2pool(output):
    # Output (n_variables x n_times)
    # Variables: (L, G, D,) = x
    # => Output [n = L+G+D]
    return [output[0]+output[1]]

def observable_2pool_2species(output):
    # Output (n_variables x n_times)
    # Variables: (L, G, D,) = x
    # => Output [n = L+G+D]
    return [output[0]+output[1],
            output[2]+output[3],
            output[0]+output[1]+output[2]+output[3]]

def observable_2pool_2species_resource(output):
    return [output[0]+output[1],
            output[2]+output[3],
            output[0]+output[1]+output[2]+output[3],
            output[4]]

def observable_2pool_2species_resource_toxin(output):
    return [output[0]+output[1],
            output[2]+output[3],
            output[4],
            output[5]]


if __name__ == "__main__":
    Nt = 1e4
    x01 = [[1e1], [0.], [0.]]
    t = np.linspace(0, 12, 200)

###############33############### 3 pool models with resource pool ##############################3########333
    # Generate model data
    const1 = [1e-2, 0., 2.5, .1, Nt] # (lambd, omega1, mu, omega2, N_t, )
    const2 = [1e-3, 0., 2.5, .1, Nt] # (lambd, omega1, mu, omega2, N_t, )
    const3 = [5e-5, 0., 2.5, .1, Nt] # (lambd, omega1, mu, omega2, N_t, )
    const4 = [1e-3, .0, 2.5, .0, Nt] # (lambd, omega1, mu, omega2, N_t, )
    const_3pool_res = [const1, const2, const3, const4]
    data_3pool_res = [generate_insilico_data(pool_model_3pools_resource, [t], [], [[]], x01,
                const=const, obs_func=observable_3pool, n_traj=1) for const in const_3pool_res]

    labels = [r"$\lambda=10^{-2}, \omega'= 0.1$", r"$\lambda=10^{-3}, \omega'= 0.1$", r"$\lambda=10^{-4}, \omega'= 0.1$", r"$\lambda=10^{-3}, \omega'= 0$"]
    color_palette_1sp = [colors_all['N_lambd_1e-3_omega_0'], colors_all['N_wo_tempshift'], colors_all['N_lambd_1e-2_omega_0'], colors_all['N_lambd_1e-3_omega_0_5']]
    lnst = ['solid', 'solid', 'solid', 'dashed']
    # Plot the model
    fig, ax = plt.subplots(1, 1, figsize=figsize_default)
    for i, data in enumerate(data_3pool_res):
        for d in data:
            ax.plot(d['times'], d['obs_mean'][0], linewidth=2.5, color=color_palette_1sp[i], label=labels[i], linestyle=lnst[i])
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacterial Count, $N$ [CFU/mL]')
    ax.set_yscale('log')
    ax.set_xlim(0., 11.0)
    #ax.set_ylim(3.0e0, 0.6e4)
    ax.legend(handlelength=1.65)
    plt.savefig('paper/Figures-pool_model_3pools_resource.pdf', bbox_inches='tight')
    plt.close(fig)

#################################### Temperature shift ##########################################
    coord_text = (0.04, 0.88)
    x01 = [[1e1], [0.], [0.], [0.]]
    Gamma, delta= 1., 5.
    shift_cnd = [(1, -5), (5, -5)] #((1, 10), (3, 5), (5, 5), (7, 15))
    # Gamma rate constants:
    sigma, delta, zeta, alpha = 2., 5., 1., 0.05
    
    const_constT = [1e-2, 0., 0.2, .1, 1e-3, Nt, sigma, delta, zeta, alpha, ()]
    #(lambd, omega1, mu, omega2, xi, N_t, sigma, delta, zeta, alpha, shift_cnd,) 
    const5 = [1e-2, 0., 0.2, .1, 1e-3, Nt, sigma, delta, zeta, alpha, shift_cnd] # (lambd, omega1, mu, omega2, xi, N_t, sigma, delta, zeta, alpha, shift_cnd,)
    const6 = [1e-2, 0., 0.2, .1, 1e-3, Nt, sigma, delta, zeta, alpha, ((1, 10), (3, -10), (7, -5))] #  (lambd, omega1, mu, omega2, xi, N_t, sigma, delta, zeta, alpha, shift_cnd,)
    
    y = [temp_stress(tt, *const5[6:])[0] for tt in t]
    y2 = [temp_stress(tt, *const6[6:])[0] for tt in t]

    consts_Tempshift = [const_constT, const5, const6]
    data_Tempshift = [generate_insilico_data(pool_model_dormant, [t], [], [[]], x01,
                                            const=const, obs_func=observable_3pool_dormant, n_traj=1) for const in consts_Tempshift]

    labels = [r'constant temperature', r'$\Delta T = -5^\circ C, -5^\circ C$',  r'$\Delta T = 10^\circ C, -10^\circ C, -5^\circ C$']
    color_palette_1sp_tempshift = [colors_all['N_wo_tempshift'], colors_all['N_tempshift_10'], colors_all['N_tempshift_10_5_15']]

    fig, ax = plt.subplots(2, 1, figsize=(6.5, 5.5), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.1)
    for i, data in enumerate(data_Tempshift):
        for d in data:
            ax[0].plot(d['times'], d['obs_mean'][0], color=color_palette_1sp_tempshift[i], label=labels[i])
    ax[1].plot(t, y, color=color_palette_1sp_tempshift[i-1])
    ax[1].plot(t, y2, color=color_palette_1sp_tempshift[i])
    fig, ax[0] = set_labels(fig, ax[0], r'', r'Bacterial Count, $N$ [CFU/mL]')
    fig, ax[1] = set_labels(fig, ax[1], r'Time, $t$ [h]', r'Rate, $\gamma$')
    ax[0].set_yscale('log')
    ax[0].set_xlim(0., 8.5)
    ax[0].set_ylim(6e0, 1.1e4)
    ax[0].legend()
    ax[0].text(8., 9.9, r'\textbf{A}')
    ax[1].text(8., 1.5,  r'\textbf{B}')
    plt.savefig('paper/Figures-pool_model_3pools_resource_tempshift.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=figsize_default)
    for i, data in enumerate(data_Tempshift):
        for d in data:
            ax.plot(d['times'], d['obs_mean'][0], color=color_palette_1sp_tempshift[i], label=labels[i])
    fig, ax = set_labels(fig, ax, r'', r'Bacterial Count, $N$ [CFU/mL]')
    ax.set_yscale('log')
    ax.set_xlim(0.0, 9)
    ax.set_ylim(6e0, 2e4)
    ax.legend()
    ax.text(0.04, 0.9, r'\textbf{A}', transform = ax.transAxes)
    plt.savefig('paper/Figures-pool_model_3pools_resource_tempshift1.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=figsize_default_small)
    ax.plot(t, y, color=color_palette_1sp_tempshift[i-1])
    ax.plot(t, y2, color=color_palette_1sp_tempshift[i])
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Rate, $\gamma$')
    ax.set_xlim(0., 9)
    ax.text(0.04, 0.8, r'\textbf{B}', transform = ax.transAxes)
    plt.savefig('paper/Figures-pool_model_3pools_resource_tempshift2.pdf', bbox_inches='tight')
    plt.close(fig)

#################################### Resource competition ################################################3
    x0_gLV = [[10.], [0.], [10.], [0.], [1.]]
    const_gLV = [.01, 3., .05, 2.5, Nt] # (lambd1, lambd2, alph1, alph2, Nt, )
    data_gLV = generate_insilico_data(pool_model_resource_comp, [np.linspace(0, 10, 100)], [], [[]], x0_gLV,
                                      const=const_gLV, n_traj=1, obs_func=observable_2pool_2species_resource)
    labels = [r'$N_A=L_A+G_A$', r'$N_B=L_B+G_B$', r'$N=N_A+N_B$', r'$R N_t$']
    color_palette_2sp = [colors_all['N_A'], colors_all['N_B'], colors_all['N'], colors_all['R']]
    lnst = ['solid', 'solid', 'dashed', 'solid']

    fig, ax = plt.subplots(1, 1, figsize=figsize_default)
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacterial Count, $N$ [CFU/mL]')
    for d in data_gLV:
        for i, obs in enumerate(d['obs_mean']):
            if i == 3:
                obs = Nt*obs
            ax.plot(d['times'], obs, label=labels[i], color=color_palette_2sp[i])
    #ax.set_yscale('log')
    ax.set_xlim(0., 8.0)
    ax.set_ylim(-100, 1.05e4)
    ax.legend()
    plt.savefig('paper/Figures-pool_model_2pools_resource_competition.pdf', bbox_inches='tight')
    plt.close(fig)

######################## Interspecies competition (Waste/inhibitor production) #############################
####################################### Toxin Production ##############################################
    x0_tox = [[10.], [0.], [10.], [0.], [1.], [0.]]
    const_tox = [.01, 3., .05, 2.5, Nt, 0.2, 0.2] # (lambd1, lambd2, alph1, alph2, Nt, k, omega)
    data_tox1 = generate_insilico_data(pool_model_2sp_comp_toxin1, [np.linspace(0, 10, 100)], [], [[]], x0_tox,
                                       const=const_tox, n_traj=1, obs_func=observable_2pool_2species_resource_toxin)
    data_tox2 = generate_insilico_data(pool_model_2sp_comp_toxin2, [np.linspace(0, 10, 100)], [], [[]], x0_tox,
                                       const=const_tox, n_traj=1, obs_func=observable_2pool_2species_resource_toxin)

    fig, ax = plt.subplots(1, 2, figsize=figsize_default2subpl)
    fig.subplots_adjust(wspace=0.25)
    labels1 = [r'$N_A=L_A+G_A$', r'$N_B=L_B+G_B$', r'$R N_t$', r'$T_B^1$']
    labels2 = [r'$N_A=L_A+G_A$', r'$N_B=L_B+G_B$', r'$R N_t$', r'$T_B^2$']
    color_palette_2sp_toxin = [colors_all['N_A'], colors_all['N_B'], colors_all['R'], colors_all['T']]
    for d in data_tox1:
        for i, obs in enumerate(d['obs_mean']):
            if i == 2:
                obs = Nt*obs
            ax[0].plot(d['times'], obs, color=color_palette_2sp_toxin[i], label=labels1[i])#, linestyle=lnst[i])
    for d in data_tox2:
        for i, obs in enumerate(d['obs_mean']):
            if i == 2:
                obs = Nt*obs
            ax[1].plot(d['times'], obs, color=color_palette_2sp_toxin[i], label=labels2[i])#, linestyle=lnst[i])
    for i in range (2):
        ax[i].set_xlim(2., 9.0)
        ax[i].set_ylim(-100, 1.01e4)
        ax[i].legend()
        fig, ax[i] = set_labels(fig, ax[i], r'Time, $t$ [h]', r'Bacterial Count, $N$ [CFU/mL]')
    ax[0].text(2.1, 800, r'\textbf{A}')#r'(a) $\mathcal{F}^1$')
    ax[1].text(2.1, 800, r'\textbf{B}')#r'(b) $\mathcal{F}^2$')
    #plt.savefig('paper/Figures-pool_model_2pools_toxin.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=figsize_default)
    for d in data_tox1:
        for i, obs in enumerate(d['obs_mean']):
            if i == 2:
                obs = Nt*obs
            ax.plot(d['times'], obs, color=color_palette_2sp_toxin[i], label=labels1[i])#, linestyle=lnst[i])
    ax.set_xlim(2., 9.0)
    ax.set_ylim(-100, 1.01e4)
    ax.legend()
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacterial Count, $N$ [CFU/mL]')
    ax.text(*coord_text, r'\textbf{A}', transform = ax.transAxes)
    plt.savefig('paper/Figures-pool_model_2pools_toxin1.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=figsize_default)
    for d in data_tox2:
        for i, obs in enumerate(d['obs_mean']):
            if i == 2:
                obs = Nt*obs
            ax.plot(d['times'], obs, color=color_palette_2sp_toxin[i], label=labels2[i])#, linestyle=lnst[i])
    ax.set_xlim(2., 9.0)
    ax.set_ylim(-100, 1.01e4)
    ax.legend()
    fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'Bacterial Count, $N$ [CFU/mL]')
    ax.text(*coord_text, r'\textbf{B}', transform = ax.transAxes)
    plt.savefig('paper/Figures-pool_model_2pools_toxin2.pdf', bbox_inches='tight')
    plt.close(fig)

############################# Inhibition ######################################
    x0_inhib = [[0.1], [0.1]]
    const_inhib = [1., 0.5, 2] # ()
    data_inhib = []
    lnst = ['solid', 'dashed', 'dotted']
    color_palette_2sp_only = [colors_all['N_A'], colors_all['N_B']]
    for psi in const_inhib:
        data_inhib += generate_insilico_data(pool_model_2sp_comp_inhib, [np.linspace(0, 10, 100)], [], [[]], x0_inhib,
                                             const=[psi], n_traj=1)

    fig, ax = plt.subplots(3, 1, figsize=(4, 6.7), sharex=True)
    fig.subplots_adjust(hspace=0.25)
    labels = [r'$G_A$', r'$G_B$']
    for j, d in enumerate(data_inhib):
        for i, obs in enumerate(d['obs_mean']):
            ax[j].plot(d['times'], obs, linewidth=3-i, color=color_palette_2sp_only[i], label=labels[i], linestyle='dashed')
        ax[j].set_xlim(-0.001, 1.)
        ax[j].legend(loc='center right')
        fig, ax[j] = set_labels(fig, ax[j], r'Time, $t$ [h]', r'$N / N_t$')
    ax[0].text(*coord_text, r'\textbf{A}', transform = ax[0].transAxes)#r'(a) $\psi$ = 1')
    ax[1].text(*coord_text, r'\textbf{B}', transform = ax[1].transAxes)#r'(b) $\psi$ = 0.5')
    ax[2].text(*coord_text, r'\textbf{C}', transform = ax[2].transAxes)#r'(c) $\psi$ = 2')
    ax[0].set_ylim(0.1, 0.48)
    ax[1].set_ylim(0.1, 0.83)
    ax[2].set_ylim(0.1, 0.87)
    #plt.savefig('paper/Figures-pool_model_1pool_inhib.pdf', bbox_inches='tight')
    plt.close(fig)

    #text = [r'(a) $\psi$ = 1', r'(b) $\psi$ = 0.5', r'(c) $\psi$ = 2']
    text = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}']
    text_coord = [(0.035, 0.435), (0.035, 0.74), (0.035, 0.78)]
    y_max_lim = [0.48, 0.83,  0.87]
    for j, d in enumerate(data_inhib):
        fig, ax = plt.subplots(1, 1, figsize=(6., 3.5))#figsize_default)
        for i, obs in enumerate(d['obs_mean']):
            ax.plot(d['times'], obs, linewidth=3-i, color=color_palette_2sp_only[i], label=labels[i], linestyle='dashed')
        ax.set_xlim(-0.001, 1.)
        ax.legend(loc='center right')
        fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'$N / N_t$')
        ax.text(*coord_text, text[j], transform = ax.transAxes)
        ax.set_ylim(0.1, y_max_lim[j])
        plt.savefig(f'paper/Figures-pool_model_1pool_inhib{j+1}.pdf', bbox_inches='tight')
        plt.close(fig)


################33 Interspecies Cooperation (Activation) ##############################3333
    
    lwdth = [3., 2., 2., 2.]
    x0_coop = [[0.1], [0.1], [0.01], [0.01]]

    fig, ax = plt.subplots(3, 1, figsize=(4, 6.7), sharex=True)
    fig.subplots_adjust(hspace=0.25)
    lnst = ['dashed', 'dashed', 'solid', 'solid']
    labels = [r'$G_A$', r'$G_B$', r'$P_A$', r'$P_B$']
    color_palette_2sp_coop = [colors_all['N_A'], colors_all['N_B'], colors_all['T_A'], colors_all['T']]
    const_coop = [
        [1., 1.], # (psi, phi)
        [1., 2.],
        [2., 1.]]
    for k, c in enumerate(const_coop):
        data_coop = generate_insilico_data(pool_model_2sp_cooper, [np.linspace(0, 10.5, 100)], [], [[]], x0_coop,
                                           const=c, n_traj=1)
        for j, d in enumerate(data_coop):
            for i, obs in enumerate(d['obs_mean']):
                ax[k].plot(d['times'], obs, color=color_palette_2sp_coop[i], label=labels[i], linestyle=lnst[i],
                           linewidth=lwdth[i])
        fig, ax[k] = set_labels(fig, ax[k], r'Time, $t$ [h]', r'$N / N_t$')
        ax[k].set_xlim(-0.001, 10.5)
    ax[0].set_ylim(0.0, 1.47)
    ax[1].set_ylim(0.0, 1.22)
    ax[2].set_ylim(0.0, 0.9)
    ax[0].text(0.05, 0.85, r'\textbf{A}', transform = ax[0].transAxes)
    ax[1].text(0.05, 0.85, r'\textbf{B}', transform = ax[1].transAxes)
    ax[2].text(0.05, 0.85, r'\textbf{C}', transform = ax[2].transAxes)
    ax[0].legend(loc='upper left', ncols=2)
    #plt.savefig('paper/Figures-pool_model_2sp_1pool_coop.pdf', bbox_inches='tight')
    plt.close(fig)

    ymax_lim = [0.9, 1.3, 0.9]
    text = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}']

    for k, c in enumerate(const_coop):
        fig, ax = plt.subplots(1, 1, figsize=(6., 3.5))#figsize_default)
        data_coop = generate_insilico_data(pool_model_2sp_cooper, [np.linspace(0, 10.5, 100)], [], [[]], x0_coop,
                                           const=c, n_traj=1)
        for j, d in enumerate(data_coop):
            for i, obs in enumerate(d['obs_mean']):
                ax.plot(d['times'], obs, color=color_palette_2sp_coop[i], label=labels[i], linestyle=lnst[i],
                           linewidth=lwdth[i])
        fig, ax = set_labels(fig, ax, r'Time, $t$ [h]', r'$N / N_t$')
        ax.set_xlim(-0.001, 10.5)
        ax.set_ylim(0.0, ymax_lim[k])
        ax.text(*coord_text, text[k], transform = ax.transAxes)
        ax.legend(loc='center left')#ncol=2, loc='lower right')
        plt.savefig(f'paper/Figures-pool_model_2sp_1pool_coop{k+1}.pdf', bbox_inches='tight')
        plt.close(fig)

