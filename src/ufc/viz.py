"""Shared matplotlib style so every notebook chart looks like one system."""
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# Fixed categorical order (validated palette). Corner colors keep their real-world meaning.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RED_CORNER = "#e34948"
BLUE_CORNER = "#2a78d6"
NEUTRAL = "#a3a29c"
METHOD_COLORS = {"KO/TKO": SERIES[1], "SUB": SERIES[2], "DEC": SERIES[0], "Other": NEUTRAL}


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.figsize": (9, 4.5), "figure.dpi": 110,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "lines.linewidth": 2, "legend.frameon": False,
    })
