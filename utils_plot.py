import matplotlib as mpl


def set_paper_style():
    """
    Configure Matplotlib global settings for consistent figure style
    across all scripts used in the paper.
    """

    # ----- PDF & PostScript: text editable -----
    mpl.rcParams["pdf.fonttype"] = 42      # TrueType, editable in Illustrator
    mpl.rcParams["ps.fonttype"] = 42

    # ----- Axes styling -----
    mpl.rcParams["axes.linewidth"] = 1.0   # thicker axis border
    mpl.rcParams["axes.labelsize"] = 6    # x/y label font size
    mpl.rcParams["axes.titlesize"] = 6    # subplot title font
    mpl.rcParams["figure.titlesize"] = 7  # suptitle font size

    # ----- Tick labels -----
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6

    # ----- Legend -----
    mpl.rcParams["legend.fontsize"] = 5
    mpl.rcParams["legend.frameon"] = False

    # ----- Figure DPI settings -----
    mpl.rcParams["savefig.dpi"] = 300      # output quality
    mpl.rcParams["figure.dpi"] = 150       # notebook display quality

    # ----- Default font -----
    # Do NOT force Arial — matplotlib chooses the best available system font
    mpl.rcParams["font.family"] = "sans-serif"

    # Optional: choose better default sans-serif list
    mpl.rcParams["font.sans-serif"] = [
        "DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"
    ]

    # ----- Other nice aesthetic tweaks -----
    mpl.rcParams["axes.spines.top"] = True
    mpl.rcParams["axes.spines.right"] = True
    mpl.rcParams["figure.autolayout"] = False   # let tight_layout handle layout

    # Optionally define a consistent colormap palette
    mpl.rcParams["image.cmap"] = "viridis"

    print("[utils_plot] Paper style has been applied.")
