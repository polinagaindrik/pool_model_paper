import matplotlib.pyplot as plt

COLORS_ALL = {
    "R": "#808080",
    "N_A": "#D06062",
    "N_B": "#4E89B1",
    "N": "#7E57A5",
    "T": "#99582A",
    "T_A": "#c79758",
    "N_lambd_1e-2_mu_0": "#E2B100",
    "N_lambd_1e-3_mu_0": "#386641",
    "N_lambd_1e-3_mu_0_5": "#0982A4",
    "N_wo_tempshift": "#679E48",
    "N_tempshift_10": "#ED733E",
    "N_tempshift_10_5_15": "#C3568A",
}

FIGSIZE_DEFAULT = (6.5, 4.0)
FIGSIZE_DEFAULT_SMALL = (6.5, 2.0)
FIGSIZE_DEFAULT_2SUBPL = (13, 4)


def load_style():
    plt.rcParams["figure.dpi"] = 400
    plt.rcParams["font.family"] = "serif"
    plt.rc("text", usetex=True)
    plt.rcParams["text.latex.preamble"] = r"\usepackage{bm} \usepackage{amsmath}"

    plt.rcParams["legend.fontsize"] = 15.0
    plt.rcParams["legend.framealpha"] = 0.0
    plt.rcParams["legend.handlelength"] = 1.8
    plt.rcParams["axes.prop_cycle"] = plt.cycler(linewidth=[2.5])
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["font.size"] = 15
    plt.rc("xtick", labelsize=13)
    plt.rc("ytick", labelsize=13)
