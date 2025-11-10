import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ------------------ Input files ------------------
FILE_QIF = "qif_parabola_result.mat"
FILES_CONV = {
    "LIF-32 (conv)":  "conversion_snn_parabola32.mat",
    "LIF-32 (cal)":   "conversion_cali_snn_parabola32.mat",
    "LIF-64 (conv)":  "conversion_snn_parabola64.mat",
    "LIF-64 (cal)":   "conversion_cali_snn_parabola64.mat",
    "LIF-128 (conv)": "conversion_snn_parabola128.mat",
    "LIF-128 (cal)":  "conversion_cali_snn_parabola128.mat",
}

# ------------------ Style ------------------
PALETTE = {
    "QIF":           "#1f6fb4",  # blue
    "LIF-32 (conv)": "#fca082",
    "LIF-32 (cal)":  "#ef6a4b",
    "LIF-64 (conv)": "#ef3b2c",
    "LIF-64 (cal)":  "#c3271b",
    "LIF-128 (conv)":"#b2182b",
    "LIF-128 (cal)": "#7f1010",
}

LINEWIDTH = {
    "QIF": 1.6,          # slightly emphasized but still thin
    "default": 1.1,      # thin solid lines for all others
}

FIGSIZE = (8.2, 4.4)     # consistent height for both figures

plt.rcParams.update({
    "figure.dpi": 160,
    "figure.autolayout": True,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
})

# ------------------ Helpers ------------------
def loadmat(path: Path):
    d = sio.loadmat(path)
    return {k: np.squeeze(np.asarray(v)) for k, v in d.items() if not k.startswith("__")}

def rel_l2_pct(y_true, y_pred):
    return 100.0 * np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1e-12)

def metric_row(model, quantity, y_true, y_pred):
    return {
        "Model": model,
        "Quantity": quantity,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "RelL2_pct": rel_l2_pct(y_true, y_pred),
    }

# ------------------ Plot (with non-overlapping inset + figure legend) ------------------
def plot_with_zoom(x, truth, preds, present_models, qkey, ylabel, zoom_range, outpath):
    # Leave room on the right for BOTH legend and inset
    # fig, ax = plt.subplots(figsize=FIGSIZE)
    # fig,ax = plt.subplots(figsize=(8.6, 5.2))
    # fig.subplots_adjust(right=0.74)   # reserve a right column
    fig = plt.figure(figsize=(8.6, 5.2))
    ax = fig.add_subplot(111)

    # --- main curves ---
    handles = []
    h_truth, = ax.plot(x, truth[qkey], color="k", lw=1.5, alpha=0.9,
                       label="Ground Truth", zorder=1)
    handles.append(h_truth)
    for model in present_models:
        color = PALETTE.get(model, "#555")
        lw = LINEWIDTH.get(model, LINEWIDTH["default"])
        h, = ax.plot(x, preds[model][qkey], color=color, lw=lw, alpha=0.95,
                     label=model, zorder=2)
        handles.append(h)

    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Predicted Parabola with Zoom (QIF vs Conversion)", fontsize=14, pad=6)
    ax.grid(alpha=0.25)

    # --- zoom window stats ---
    x_min, x_max = zoom_range
    mask = (x >= x_min) & (x <= x_max)
    y_stack = np.vstack([truth[qkey][mask]] + [preds[m][qkey][mask] for m in present_models])
    y_min, y_max = float(np.min(y_stack)), float(np.max(y_stack))
    pad_y = 0.05 * (y_max - y_min + 1e-12)

    # rectangle on main axes
    rect = Rectangle((x_min, y_min - pad_y),
                     x_max - x_min, (y_max - y_min) + 2*pad_y,
                     fill=False, lw=1.1, linestyle="--", edgecolor="gray", alpha=0.9)
    ax.add_patch(rect)

    # --- legend OUTSIDE (top of right column) ---
    # place legend in the reserved right gutter; then inset goes below it
    # fig.legend(handles=handles,
    #            labels=[h.get_label() for h in handles],
    #            loc="center", bbox_to_anchor=(0.985, 0.98),
    #            frameon=False, ncol=1)
    ax.legend(fontsize=9, loc="center", bbox_to_anchor=(0.5, 0.5))

    # --- inset OUTSIDE (right column, below the legend) ---
    # The bbox spans the right gutter (x0,y0,w,h in axes fraction of 'ax')
    axins = inset_axes(ax, width="42%", height="42%",
                       bbox_to_anchor=(1.02, 0.20, 0.42, 0.46),   # lower on right side
                       bbox_transform=ax.transAxes, loc="upper left", borderpad=0.0)

    # inset curves (no legend)
    axins.plot(x[mask], truth[qkey][mask], color="k", lw=1.2, alpha=0.9)
    for model in present_models:
        color = PALETTE.get(model, "#555")
        lw = LINEWIDTH.get(model, LINEWIDTH["default"])
        axins.plot(x[mask], preds[model][qkey][mask], color=color, lw=lw, alpha=0.95)

    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min - pad_y, y_max + pad_y)
    axins.tick_params(axis="both", labelsize=9)
    axins.set_title("Zoom", fontsize=9, pad=2)

    # --- diagonal connectors (Figure 3 style) ---
    corners = [(x_min, y_min - pad_y),
               (x_max, y_min - pad_y),
               (x_min, y_max + pad_y),
               (x_max, y_max + pad_y)]
    # map all to lower-left corner of inset for a clean look
    xB, yB = axins.get_xlim()[0], axins.get_ylim()[0]
    for (xA, yA) in corners:
        con = ConnectionPatch(xyA=(xA, yA), coordsA=ax.transData,
                              xyB=(xB, yB), coordsB=axins.transData,
                              color="gray", lw=0.8, alpha=0.8)
        fig.add_artist(con)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

# ------------------ Main ------------------
def main(outdir="."):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    saved = []

    # ---- Load QIF (provides x & ground truth) ----
    if not Path(FILE_QIF).exists():
        raise FileNotFoundError(f"Missing required file: {FILE_QIF}")
    qif = loadmat(Path(FILE_QIF))
    for key in ["u_true", "u_pred"]:
        if key not in qif:
            raise KeyError(f"{FILE_QIF} missing key: {key}")

    # x = np.ravel(qif["x"])
    x = np.linspace(-1,1,1000)
    truth = {
        "u":   np.ravel(qif["u_true"])
    }
    preds = {
        "QIF": {
            "u":   np.ravel(qif["u_pred"])
        }
    }
    present_models = ["QIF"]

    # ---- Load conversion (uncalibrated + calibrated) ----
    for label, fn in FILES_CONV.items():
        p = Path(fn)
        if not p.exists():
            print(f"Warning: missing {fn}; skipping {label}")
            continue
        d = loadmat(p)
        for need in ["u_pred"]:
            if need not in d:
                print(f"Warning: {fn} missing {need}; skipping {label}")
                break
        else:
            preds[label] = {"u": np.ravel(d["u_pred"])}
            present_models.append(label)

    # ---- METRICS (like parabola): MAE, RMSE, R2, RelL2% ----
    rows = []
    for model in present_models:
        for q in ["u"]:
            rows.append(metric_row(model, q, truth[q], preds[model][q]))
    metrics_df = pd.DataFrame(rows).set_index(["Model", "Quantity"]).sort_index()
    metrics_csv = outdir / "parabola_metrics_u.csv"
    metrics_df.to_csv(metrics_csv)
    saved.append(metrics_csv)

    with pd.option_context("display.precision", 6):
        print("\n=== PINN-1D Metrics (u) ===")
        print(metrics_df, "\n")

    # ---- PLOTS (separate figures, thin lines, zoom in) ----
    plot_with_zoom(x, truth, preds, present_models,
                   qkey="u", ylabel="u(x)", zoom_range=(0.7, 0.75),
                   outpath=outdir / "parabola_u_zoom.png")
    saved.append(outdir / "parabola_u_zoom.png")



    # ---- report outputs ----
    print("Saved files:")
    for p in saved:
        print("  -", Path(p).resolve())

if __name__ == "__main__":
    main(".")
