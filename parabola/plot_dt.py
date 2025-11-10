import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# ------------------ File map ------------------
DATAFILES = {
    "QIF": "qif_parabola_result.mat",
    "LIF-32": "dt_parabola_result32.mat",
    "LIF-64": "dt_parabola_result64.mat",
    "LIF-128": "dt_parabola_result128.mat",
}

PALETTE = {
    "QIF":     "#1f6fb4",  # strong blue
    "LIF-32":  "#fca082",  # light warm
    "LIF-64":  "#ef3b2c",  # medium warm
    "LIF-128": "#7f1010",  # deep warm
}
MARKER = "s"  # small squares for LIF to highlight jaggedness
# PRED_LINESTYLE = (0, (4, 3))   # dashed line pattern (on 4, off 3)
PRED_LINEWIDTH_QIF = 1.2
PRED_LINEWIDTH_LIF = 1.2

plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
})

# ------------------ IO helpers ------------------
def load_mat(path: Path):
    d = sio.loadmat(path)
    d = {k: v for k, v in d.items() if not k.startswith("__")}
    sq = lambda x: np.squeeze(np.asarray(x))
    return {"y_true": sq(d["y_true"]), "y_pred": sq(d["y_pred"]),
            "time": sq(d["time"]), "loss": sq(d["loss"])}

def metrics(y_true, y_pred):
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    rel_l2 = 100*np.linalg.norm(y_true - y_pred)/(np.linalg.norm(y_true)+1e-12)
    return dict(MAE=mae, RMSE=rmse, R2=r2, RelL2_pct=rel_l2)

# ------------------ Main ------------------
def main(outdir="."):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    saved = []

    # x is fixed by you
    x = np.linspace(-1, 1, 1000)

    # Load data
    results, missing = {}, []
    for name, fn in DATAFILES.items():
        p = Path(fn)
        if p.exists():
            results[name] = load_mat(p)
        else:
            missing.append(fn)
    if not results:
        raise FileNotFoundError("No .mat files found. Expected: " + ", ".join(DATAFILES.values()))
    if missing:
        print("Warning: missing files (skipped):", ", ".join(missing))

    # Use ground-truth from any available file
    y_true = results[next(iter(results))]["y_true"].ravel()

    # ---------------- Metrics CSV + console print ----------------
    rows = []
    for name, d in results.items():
        rows.append({"Model": name, **metrics(d["y_true"], d["y_pred"])})
    metrics_df = pd.DataFrame(rows).set_index("Model").sort_index()
    metrics_csv = outdir / "parabola_metrics.csv"
    metrics_df.to_csv(metrics_csv); saved.append(metrics_csv)
    with pd.option_context("display.precision", 6):
        print("\n=== Regression Metrics ===")
        print(metrics_df, "\n")

    # ---------------- Plot: Loss (log-y) ----------------
    # fig = plt.figure(figsize=(6.6, 4.2))
    fig = plt.figure(figsize=(8.6, 5.2))
    for name, d in results.items():
        ep = np.arange(1, len(d["loss"]) + 1)
        lw = 3.0 if name == "QIF" else 1.6
        plt.plot(ep, d["loss"], lw=lw, color=PALETTE[name], label=name, zorder=3)
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Loss (log)")
    plt.title("Training Loss vs Epoch (QIF vs LIF)"); plt.legend(ncol=2)
    p = outdir / "parabola_loss_log.png"; plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)

    # ---------------- Time: compile / per-epoch / cumulative ----------------
    compile_time, train_times = {}, {}
    for name, d in results.items():
        t = np.ravel(d["time"])
        if len(t) >= 2:
            compile_time[name], train_times[name] = float(t[0]), t[1:]
        else:
            compile_time[name], train_times[name] = 0.0, t

    labels = list(results.keys()); colors = [PALETTE[k] for k in labels]

    # compile bar
    fig = plt.figure(figsize=(6.0, 3.6))
    plt.bar(labels, [compile_time[k] for k in labels], color=colors)
    plt.ylabel("Compile time (s)"); plt.title("Compilation Time (Epoch 0)")
    p = outdir / "parabola_time_compile.png"; plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)

    # per-epoch
    fig = plt.figure(figsize=(6.6, 4.2))
    for name in labels:
        t = np.asarray(train_times[name]); ep = np.arange(1, len(t)+1)
        if len(ep): plt.plot(ep, t, lw=3.0 if name=="QIF" else 1.6, color=PALETTE[name], label=name)
    plt.xlabel("Epoch"); plt.ylabel("Time per epoch (s)"); plt.title("Training Time per Epoch")
    plt.legend(ncol=2)
    p = outdir / "parabola_time_per_epoch.png"; plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)

    # cumulative
    fig = plt.figure(figsize=(6.6, 4.2))
    for name in labels:
        cum = np.cumsum(train_times[name]); ep = np.arange(1, len(cum)+1)
        if len(ep): plt.plot(ep, cum, lw=3.0 if name=="QIF" else 1.8, color=PALETTE[name], label=name)
    plt.xlabel("Epoch"); plt.ylabel("Cumulative time (s)"); plt.title("Cumulative Training Time (no compile)")
    plt.legend(ncol=2)
    p = outdir / "parabola_time_cumulative.png"; plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)

    # ---------------- Pred vs Truth scatter (kept) ----------------
    fig = plt.figure(figsize=(6.0, 6.0))
    all_true = np.concatenate([results[k]["y_true"].ravel() for k in results])
    lo, hi = all_true.min(), all_true.max()
    pad = 0.03*(hi - lo + 1e-12)
    plt.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", lw=1.2, alpha=0.7, label="y = x")
    rng = np.random.default_rng(0)
    for name, d in results.items():
        yt, yp = np.ravel(d["y_true"]), np.ravel(d["y_pred"])
        idx = rng.choice(len(yt), size=min(3000, len(yt)), replace=False)
        plt.scatter(yt[idx], yp[idx], s=8, alpha=0.35, color=PALETTE[name], label=name)
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("Prediction vs Truth")
    plt.legend(ncol=2)
    p = outdir / "parabola_pred_vs_true.png"; plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)

    # ---------------- Main + Zoom (inset OUTSIDE, no covering) ----------------
    fig = plt.figure(figsize=(8.6, 5.2))
    ax = fig.add_subplot(111)

    # main curves
    ax.plot(x, y_true, color="k", lw=2.4, alpha=0.85, label="Ground Truth", zorder=1)
    if "QIF" in results:
        ax.plot(x, results["QIF"]["y_pred"].ravel(),
            color=PALETTE["QIF"],
            lw=PRED_LINEWIDTH_QIF,
            alpha=0.9,
            label="QIF",
            zorder=2)

    # LIFs (dashed thin)
    for name in ["LIF-32", "LIF-64", "LIF-128"]:
        if name in results:
            y = results[name]["y_pred"].ravel()
            ax.plot(x, y,
                    color=PALETTE[name],
                    lw=PRED_LINEWIDTH_LIF,
                    alpha=0.95,
                    label=name,
                    zorder=3)
            # step = max(1, len(x)//70)
            # ax.plot(x[::step], y[::step], linestyle="None", marker=MARKER, ms=3.6,
            #         mec=PALETTE[name], mfc="white", alpha=0.9, zorder=6)

    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title("Predicted Parabola with Zoom (QIF vs LIF)")
    ax.legend(ncol=3)

    # choose zoom range (edit as desired)
    # x_min, x_max = -0.12, 0.12
    x_min, x_max = 0.3, 0.4
    mask = (x >= x_min) & (x <= x_max)

    # compute y-limits from ALL curves (so inset not clipped)
    ys_in_window = [y_true[mask]]
    if "QIF" in results: ys_in_window.append(results["QIF"]["y_pred"].ravel()[mask])
    for name in ["LIF-32", "LIF-64", "LIF-128"]:
        if name in results: ys_in_window.append(results[name]["y_pred"].ravel()[mask])
    y_stack = np.vstack(ys_in_window)
    y_min, y_max = float(np.min(y_stack)), float(np.max(y_stack))
    pad_y = 0.05 * (y_max - y_min + 1e-12)

    # draw zoom rectangle on main axes
    rect = Rectangle((x_min, y_min - pad_y), x_max - x_min, (y_max - y_min) + 2*pad_y,
                     fill=False, lw=1.5, linestyle="--", edgecolor="gray", alpha=0.9, zorder=7)
    ax.add_patch(rect)

    # inset placed OUTSIDE the main axes (right side) so it never covers the curves
    axins = inset_axes(ax, width="42%", height="46%",
                       bbox_to_anchor=(1.04, 0.52, 0.42, 0.46),  # (x0,y0,w,h) in axes fraction space; anchored at left
                       bbox_transform=ax.transAxes, loc="center left", borderpad=0.0)

    # inset curves
    axins.plot(x[mask], y_true[mask], color="k", lw=2.2, alpha=0.85)
    if "QIF" in results:
        axins.plot(x[mask], results["QIF"]["y_pred"].ravel()[mask], color=PALETTE["QIF"], lw=3.0)
    for name in ["LIF-32", "LIF-64", "LIF-128"]:
        if name in results:
            xs = x[mask]; ys = results[name]["y_pred"].ravel()[mask]
            axins.plot(xs, ys, color=PALETTE[name], lw=1.6, alpha=0.95)
            # step_z = max(1, len(xs)//30)
            # axins.plot(xs[::step_z], ys[::step_z], linestyle="None", marker=MARKER, ms=4.0,
            #            mec=PALETTE[name], mfc="white", alpha=0.95)

    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min - pad_y, y_max + pad_y)
    axins.tick_params(axis="both", labelsize=9)
    axins.set_title("Zoom", fontsize=10, pad=2)

    # connectors between rectangle and inset (since inset is outside, use ConnectionPatch)
    for (x0, y0) in [(x_min, y_min - pad_y), (x_max, y_min - pad_y),
                     (x_min, y_max + pad_y), (x_max, y_max + pad_y)]:
        con = ConnectionPatch(xyA=(x0, y0), coordsA=ax.transData,
                              xyB=(axins.get_xlim()[0], axins.get_ylim()[0]), coordsB=axins.transData,
                              color="gray", lw=0.8, alpha=0.8)
        fig.add_artist(con)

    out_zoom = outdir / "parabola_with_zoom.png"
    fig.savefig(out_zoom, dpi=300)
    plt.close(fig)
    saved.append(out_zoom)

    # ---------------- Print saved files ----------------
    print("Saved files:")
    for p in saved:
        print("  -", p.resolve())

if __name__ == "__main__":
    main(".")