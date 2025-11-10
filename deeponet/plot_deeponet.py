import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------ File map ------------------
# Note: handle the common filename typo for 128 ("deeonet" vs "deeponet") below
DATAFILES = {
    "QIF":     "experiments/deeponet/qif_deeponet_result.mat",
    "LIF-32":  "dt_deeponet32.mat",
    "LIF-64":  "dt_deeponet64.mat",
    "LIF-128": "dt_deeonet128.mat",   # if this doesn't exist, we try "dt_deeponet128.mat"
}

PALETTE = {
    "QIF":     "#1f6fb4",  # strong blue
    "LIF-32":  "#fca082",  # light warm
    "LIF-64":  "#ef3b2c",  # medium warm
    "LIF-128": "#7f1010",  # deep warm
}

# Style: predictions dashed & thin; truth solid & thicker
PRED_LINESTYLE = (0, (4, 3))   # dashed pattern (on 4, off 3)
PRED_LINEWIDTH_QIF = 2.0
PRED_LINEWIDTH_LIF = 1.2
TRUTH_LINEWIDTH = 2.4

# Figure sizing
FIGSIZE = (6.6, 4.2)

plt.rcParams.update({
    "figure.dpi": 160,
    "figure.autolayout": True,
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
})

# ------------------ IO helpers ------------------
def _maybe_load(path: Path):
    if path.exists():
        return sio.loadmat(path)
    # handle 128 typo variant
    if path.name == "dt_deeonet128.mat":
        alt = path.with_name("dt_deeponet128.mat")
        if alt.exists():
            return sio.loadmat(alt)
    return None

def load_mat_any(path: Path):
    d_raw = _maybe_load(path)
    if d_raw is None:
        raise FileNotFoundError(str(path))
    d = {k: v for k, v in d_raw.items() if not k.startswith("__")}
    sq = lambda x: np.squeeze(np.asarray(x))
    out = {}
    for key in ["y_true", "y_pred", "time", "loss", "x"]:
        if key in d:
            out[key] = sq(d[key])
    # ensure time/loss are 1D arrays
    if "time" in out: out["time"] = np.ravel(out["time"])
    if "loss" in out: out["loss"] = np.ravel(out["loss"])
    return out

def ensure_2d(arr):
    """Return (N, T): if 1D -> (1, T); if 2D keep; else flatten last dim."""
    A = np.asarray(arr)
    if A.ndim == 1:
        return A[None, :]
    if A.ndim == 2:
        return A
    # fallback: flatten trailing dims
    return A.reshape(A.shape[0], -1)

def regression_metrics(y_true_1d, y_pred_1d):
    # 1D vectors
    y_true = np.asarray(y_true_1d).ravel()
    y_pred = np.asarray(y_pred_1d).ravel()
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    rel_l2 = 100 * np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1e-12)
    return dict(MAE=mae, RMSE=rmse, R2=r2, RelL2_pct=rel_l2)

def summary_from_series(values):
    v = np.asarray(values).astype(float)
    return {
        "count": v.size,
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "q25": float(np.percentile(v, 25.0)),
        "median": float(np.percentile(v, 50.0)),
        "q75": float(np.percentile(v, 75.0)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }

# ------------------ Main ------------------
def main(outdir=".",
         n_curve_samples=5,
         zoom_x=(-0.12, 0.12),
         seed=0):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    saved = []

    # x: use from dt_deeponet32 if available; else fallback to linspace
    x = None
    # Load data
    results, missing = {}, []
    for name, fn in DATAFILES.items():
        try:
            d = load_mat_any(Path(fn))
            results[name] = d
            if x is None and "x" in d:
                x = np.ravel(d["x"])
        except FileNotFoundError:
            missing.append(fn)
    if not results:
        raise FileNotFoundError("No .mat files found. Expected: " + ", ".join(DATAFILES.values()))
    if missing:
        print("Warning: missing files (skipped):", ", ".join(missing))
    if x is None:
        # fallback (assume 1D Poisson output along a line/slice)
        any_key = next(iter(results))
        T = ensure_2d(results[any_key]["y_true"]).shape[1]
        x = np.linspace(-1, 1, T)
    x = np.ravel(x)

    # standardize shapes to (N, T)
    for k in results:
        results[k]["y_true"] = ensure_2d(results[k]["y_true"])
        results[k]["y_pred"] = ensure_2d(results[k]["y_pred"])

    # ---------------- Metrics (per-sample + summary) ----------------
    # We compute metrics per sample (row wise), then aggregate.
    rng = np.random.default_rng(seed)
    per_sample_records = []
    for name, d in results.items():
        Yt = results[name]["y_true"]
        Yp = results[name]["y_pred"]
        # If ground truth differs across models, we use each model's provided y_true.
        # (If they'd be identical, this still works.)
        for i in range(Yt.shape[0]):
            m = regression_metrics(Yt[i], Yp[i])
            m["Model"] = name
            m["Sample"] = int(i)
            per_sample_records.append(m)
    per_df = pd.DataFrame(per_sample_records).set_index(["Model", "Sample"]).sort_index()
    per_csv = outdir / "deeponet_poisson_metrics_per_sample.csv"
    per_df.to_csv(per_csv); saved.append(per_csv)

    # Make summary (mean/std/quartiles) over samples for each model
    summary_rows = []
    for name in results.keys():
        sub = per_df.loc[name]
        for metric_col in ["MAE", "RMSE", "R2", "RelL2_pct"]:
            stats = summary_from_series(sub[metric_col].values)
            stats["Model"] = name
            stats["Metric"] = metric_col
            summary_rows.append(stats)
    summary_df = pd.DataFrame(summary_rows).set_index(["Model", "Metric"]).sort_index()
    summary_csv = outdir / "deeponet_poisson_metrics_summary.csv"
    summary_df.to_csv(summary_csv); saved.append(summary_csv)

    # Also print a compact summary
    print("\n=== Metrics Summary (mean ± std; median [q25, q75]) ===")
    for name in results.keys():
        row_MAE = summary_df.loc[(name, "MAE")]
        row_RMSE = summary_df.loc[(name, "RMSE")]
        row_R2 = summary_df.loc[(name, "R2")]
        row_Rel = summary_df.loc[(name, "RelL2_pct")]
        print(f"{name:>8} | "
              f"MAE {row_MAE['mean']:.4g} ± {row_MAE['std']:.4g} "
              f"(med {row_MAE['median']:.4g} [{row_MAE['q25']:.4g},{row_MAE['q75']:.4g}]); "
              f"RMSE {row_RMSE['mean']:.4g} ± {row_RMSE['std']:.4g}; "
              f"R2 {row_R2['mean']:.4g} ± {row_R2['std']:.4g}; "
              f"RelL2% {row_Rel['mean']:.4g} ± {row_Rel['std']:.4g}")
    print()

    # ---------------- Plot: Loss (log-y) ----------------
    fig = plt.figure(figsize=FIGSIZE)
    for name, d in results.items():
        loss = np.ravel(d["loss"])
        ep = np.arange(1, len(loss) + 1)
        lw = 3.0 if name == "QIF" else 1.6
        plt.plot(ep, loss, lw=lw, color=PALETTE[name], label=name, zorder=3)
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Loss (log)")
    plt.title("Training Loss vs Epoch (DeepONet Poisson)"); plt.legend(ncol=2)
    p = outdir / "deeponet_poisson_loss_log.png"; plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)

    # ---------------- Time (BATCHED: 16 per epoch, first 16 = compile) ----------------
    B = 16  # number of batches per epoch

    def split_time_batches_first16_compile(t, B=16):
        """
        Interpret the first 16 entries as compilation (warmup) time,
        the remaining entries as per-batch times for training.
        """
        t = np.ravel(t).astype(float)
        n = len(t)
        if n <= B:
            # fallback: everything is compile
            return float(np.sum(t)), np.zeros((0, B))
        compile_time = float(np.sum(t[:B]))
        remaining = t[B:]
        # pad/truncate to multiple of B
        m = len(remaining)
        trim = m - (m // B) * B
        if trim != 0:
            remaining = remaining[:m - trim]
        batches_2d = remaining.reshape(-1, B)
        return compile_time, batches_2d

    compile_time, train_batches = {}, {}
    for name, d in results.items():
        ct, bt = split_time_batches_first16_compile(d["time"], B=B)
        compile_time[name] = ct
        train_batches[name] = bt  # shape: (n_epochs, B)

    labels = list(results.keys())
    colors = [PALETTE[k] for k in labels]

    # (1) Compile time bar
    fig = plt.figure(figsize=FIGSIZE)
    vals = [compile_time[k] for k in labels]
    plt.bar(labels, vals, color=colors)
    plt.ylabel("Compile time (s)")
    plt.title(f"Compilation Time (Epoch 0)")
    p = outdir / "deeponet_poisson_time_compile.png"
    plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)

    # (2) Per-epoch training time (sum over batches)
    fig = plt.figure(figsize=FIGSIZE)
    for name in labels:
        bt = train_batches[name]
        if bt.size == 0:
            continue
        per_epoch = bt.sum(axis=1)
        ep = np.arange(1, len(per_epoch)+1)
        plt.plot(ep, per_epoch, lw=3.0 if name == "QIF" else 1.6,
                color=PALETTE[name], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Training time per epoch (s)")
    plt.title(f"Training Time per Epoch (sum of {B} batches)")
    plt.legend(ncol=2)
    p = outdir / "deeponet_poisson_time_per_epoch.png"
    plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)

    # (3) Per-batch time profile (average across epochs)
    fig = plt.figure(figsize=FIGSIZE)
    batch_idx = np.arange(1, B+1)
    for name in labels:
        bt = train_batches[name]
        if bt.size == 0:
            continue
        mean_per_batch = bt.mean(axis=0)
        std_per_batch = bt.std(axis=0, ddof=1) if bt.shape[0] > 1 else np.zeros_like(mean_per_batch)
        plt.plot(batch_idx, mean_per_batch, lw=2.0 if name=="QIF" else 1.4,
                color=PALETTE[name], label=name)
        plt.fill_between(batch_idx, mean_per_batch - std_per_batch,
                        mean_per_batch + std_per_batch,
                        color=PALETTE[name], alpha=0.15, linewidth=0)
    plt.xlabel("Batch index (1 … 16)")
    plt.ylabel("Avg time per batch (s)")
    plt.title("Per-batch Time Profile (mean ± std across epochs)")
    plt.legend(ncol=2)
    p = outdir / "deeponet_poisson_time_per_batch.png"
    plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)

    # (4) Cumulative training time (exclude compile)
    fig = plt.figure(figsize=FIGSIZE)
    for name in labels:
        bt = train_batches[name]
        if bt.size == 0:
            continue
        per_epoch = bt.sum(axis=1)
        cum = np.cumsum(per_epoch)
        ep = np.arange(1, len(cum)+1)
        plt.plot(ep, cum, lw=3.0 if name=="QIF" else 1.8,
                color=PALETTE[name], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative time (s)")
    plt.title("Cumulative Training Time (no compile)")
    plt.legend(ncol=2)
    p = outdir / "deeponet_poisson_time_cumulative.png"
    plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)


    # ---------------- Pred vs Truth scatter (across all samples) ----------------
    # sample uniformly across models to avoid overload; still informative
    fig = plt.figure(figsize=FIGSIZE)
    rng = np.random.default_rng(seed)
    for name, d in results.items():
        Yt = results[name]["y_true"]; Yp = results[name]["y_pred"]
        # flatten all points across samples
        yt_all = Yt.reshape(-1)
        yp_all = Yp.reshape(-1)
        n = yt_all.size
        # subsample up to 4000 for clarity
        take = min(4000, n)
        idx = rng.choice(n, size=take, replace=False)
        plt.scatter(yt_all[idx], yp_all[idx], s=8, alpha=0.30, color=PALETTE[name], label=name)
    # identity line
    lo = min([results[k]["y_true"].min() for k in results])
    hi = max([results[k]["y_true"].max() for k in results])
    pad = 0.03*(hi - lo + 1e-12)
    plt.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", lw=1.0, alpha=0.6, label="y = x")
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("Prediction vs Truth (all samples)")
    plt.legend(ncol=2)
    p = outdir / "deeponet_poisson_pred_vs_true.png"; plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)

    # ---------------- Boxplot: consistent color per model (box = whiskers = caps) ----------------
    fig = plt.figure(figsize=FIGSIZE)

    # lock model order to your palette order; include only those that exist in per_df
    desired_order = ["QIF", "LIF-32", "LIF-64", "LIF-128"]
    model_order = [m for m in desired_order if (m in per_df.index.levels[0])]
    data_for_box = [per_df.loc[m]["RMSE"].values for m in model_order]
    box_colors   = [PALETTE[m] for m in model_order]

  
    bp = plt.boxplot(
        data_for_box,
        labels=model_order,
        patch_artist=True,           # so boxes can be filled
        showmeans=True, meanline=True,
        medianprops=dict(color="black", linewidth=1.6),
        meanprops=dict(color="#7a7a7a", linewidth=1.6),
        whiskerprops=dict(linewidth=1.5),   # color set per-whisker below
        capprops=dict(linewidth=1.5),       # color set per-cap below
        flierprops=dict(marker="o", markersize=4, markerfacecolor="white",
                        markeredgecolor="black", linestyle="none", alpha=0.9)
    )

    # apply SAME color to each model's box + its two whiskers + its two caps
    for i, color in enumerate(box_colors):
        # box
        bx = bp["boxes"][i]
        # color = box_colors[1]
        color = "#b0b0b0"
        bx.set_facecolor(color)
        bx.set_edgecolor(color)
        bx.set_alpha(0.35)
        bx.set_linewidth(1.8)

        # whiskers (2) and caps (2) for the i-th box
        wL = bp["whiskers"][2*i]; wR = bp["whiskers"][2*i+1]
        cL = bp["caps"][2*i];     cR = bp["caps"][2*i+1]
        for obj in (wL, wR, cL, cR):
            obj.set_color(color)
            obj.set_linewidth(1.5)

    plt.ylabel("RMSE")
    plt.title("Per-sample RMSE (Boxplot)")
    plt.grid(axis="y", alpha=0.3)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.6, label="Median"),
        Line2D([0], [0], color="#7a7a7a", lw=1.6, linestyle = "--", label="Mean"),
        Line2D([0], [0], marker="o", color="black", lw=0,
            markerfacecolor="white", markeredgecolor="black",
            markersize=5, label="Outliers"),
    ]
    plt.legend(handles=legend_handles, loc="upper right", frameon=False)

    p = outdir / "deeponet_poisson_boxplot_rmse.png"
    plt.savefig(p, dpi=300); plt.close(fig); saved.append(p)



    # ---------------- Curve plots: multiple samples per model ----------------
    # pick up to n_curve_samples indices (common across models if possible)
    # we derive a set of sample indices based on the *smallest* number of samples across models
    min_samples = min(results[name]["y_true"].shape[0] for name in results)
    ns = min(n_curve_samples, min_samples)
    # sample_idxs = np.arange(min_samples) if min_samples <= ns else np.sort(np.unique(np.floor(
    #     np.linspace(0, min_samples-1, ns)).astype(int)))
    sample_idxs = [151, 304, 401, 501, 701]

    # MAIN curves (many samples overlay) + inset zoom (outside)
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)

    # Choose one model's y_true for plotting ground truth curves; if they differ, still OK (we'll use QIF if present)
    truth_key = "QIF" if "QIF" in results else next(iter(results))
    YT_truth = results[truth_key]["y_true"]

    # plot multiple ground-truth samples (solid black, thinner than parabola case if crowded)
    for i in sample_idxs:
        yt = YT_truth[i]
        ax.plot(x, yt, color="k", lw=TRUTH_LINEWIDTH, alpha=0.35 if len(sample_idxs) > 3 else 0.7,
                label="Ground Truth" if i == sample_idxs[0] else "_nolegend_", zorder=1)

    # predictions dashed & thin
    if "QIF" in results:
        for i in sample_idxs:
            yq = results["QIF"]["y_pred"][i]
            ax.plot(x, yq, color=PALETTE["QIF"], lw=PRED_LINEWIDTH_QIF, linestyle=PRED_LINESTYLE,
                    alpha=0.9, label="QIF" if i == sample_idxs[0] else "_nolegend_", zorder=2)

    for name in ["LIF-32", "LIF-64", "LIF-128"]:
        if name in results:
            for i in sample_idxs:
                yl = results[name]["y_pred"][i]
                ax.plot(x, yl, color=PALETTE[name], lw=PRED_LINEWIDTH_LIF, linestyle=PRED_LINESTYLE,
                        alpha=0.95, label=name if i == sample_idxs[0] else "_nolegend_", zorder=3)

    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title(f"DeepONet Poisson: {len(sample_idxs)} Sample Curves (dashed = predictions)")

    ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0,
    frameon=False,
    ncol=1
    )

    # # zoom region
    # x_min, x_max = zoom_x
    # mask = (x >= x_min) & (x <= x_max)

    # # compute y-limits in zoom window across all plotted curves
    # y_windows = []
    # for i in sample_idxs:
    #     y_windows.append(YT_truth[i][mask])
    #     if "QIF" in results: y_windows.append(results["QIF"]["y_pred"][i][mask])
    #     for name in ["LIF-32", "LIF-64", "LIF-128"]:
    #         if name in results:
    #             y_windows.append(results[name]["y_pred"][i][mask])
    # y_stack = np.vstack([np.ravel(v) for v in y_windows])
    # y_min, y_max = float(np.min(y_stack)), float(np.max(y_stack))
    # pad_y = 0.05 * (y_max - y_min + 1e-12)

    # rect = Rectangle((x_min, y_min - pad_y), x_max - x_min, (y_max - y_min) + 2*pad_y,
    #                  fill=False, lw=1.5, linestyle="--", edgecolor="gray", alpha=0.9, zorder=7)
    # ax.add_patch(rect)

    # inset outside main axes
    # axins = inset_axes(ax, width="42%", height="46%",
    #                    bbox_to_anchor=(1.04, 0.52, 0.42, 0.46),
    #                    bbox_transform=ax.transAxes, loc="center left", borderpad=0.0)

    # # plot same set in inset
    # for i in sample_idxs:
    #     axins.plot(x[mask], YT_truth[i][mask], color="k", lw=TRUTH_LINEWIDTH,
    #                alpha=0.35 if len(sample_idxs) > 3 else 0.7)
    # if "QIF" in results:
    #     for i in sample_idxs:
    #         axins.plot(x[mask], results["QIF"]["y_pred"][i][mask], color=PALETTE["QIF"],
    #                    lw=PRED_LINEWIDTH_QIF, linestyle=PRED_LINESTYLE, alpha=0.9)
    # for name in ["LIF-32", "LIF-64", "LIF-128"]:
    #     if name in results:
    #         for i in sample_idxs:
    #             axins.plot(x[mask], results[name]["y_pred"][i][mask], color=PALETTE[name],
    #                        lw=PRED_LINEWIDTH_LIF, linestyle=PRED_LINESTYLE, alpha=0.95)

    # axins.set_xlim(x_min, x_max)
    # axins.set_ylim(y_min - pad_y, y_max + pad_y)
    # axins.tick_params(axis="both", labelsize=9)
    # axins.set_title("Zoom", fontsize=10, pad=2)

    # # connectors (from rectangle to inset lower-left corner)
    # con = ConnectionPatch(xyA=(x_min, y_min - pad_y), coordsA=ax.transData,
    #                       xyB=(axins.get_xlim()[0], axins.get_ylim()[0]), coordsB=axins.transData,
    #                       color="gray", lw=0.8, alpha=0.8)
    # fig.add_artist(con)

    curve_out = outdir / f"deeponet_poisson_curves_{len(sample_idxs)}samples.png"
    fig.savefig(curve_out, dpi=300); plt.close(fig); saved.append(curve_out)

    # ---------------- Print saved files ----------------
    print("Saved files:")
    for p in saved:
        print("  -", p.resolve())

if __name__ == "__main__":
    # You can tweak n_curve_samples and zoom_x as you like.
    main(outdir=".", n_curve_samples=5, zoom_x=(-0.8, -0.7), seed=0)
