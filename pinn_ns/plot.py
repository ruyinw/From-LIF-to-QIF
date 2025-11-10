import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import scipy.io as sio
from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================
FILES_TRUTH = {
    "mat": "qif_pinn_ns_hard.mat",
    "x":   "x",
    "y":   "y",
    "t":   "t",
    "u":   "u_true",   # change if needed
    "v":   "v_true",
}

# Add/remove models as needed
FILES_MODELS = {
    "QIF":             "qif_pinn_ns_hard.mat",
    "LIF-32 (conv)":   "conversion_snn_pinn_ns32.mat",
    "LIF-32 (cal)":    "conversion_cali_snn_pinn_ns32.mat",
    "LIF-64 (conv)":   "conversion_snn_pinn_ns64.mat",
    "LIF-64 (cal)":    "conversion_cali_snn_pinn_ns64.mat",
    "LIF-128 (conv)":  "conversion_snn_pinn_ns128.mat",
    "LIF-128 (cal)":   "conversion_cali_snn_pinn_ns128.mat",
}

# In each model .mat, try these keys (first match wins)
PRED_KEYS = {
    "u": ["u_pred", "u", "U"],
    "v": ["v_pred", "v", "V"],
}

# Subsampling for 3D scatters
SUB_T, SUB_X, SUB_Y = 4, 4, 4

# Fixed color limits for 3D comparisons (set to None to infer from data)
U_VMIN, U_VMAX = None, None
V_VMIN, V_VMAX = None, None

# Snapshot times to visualize (either absolute times in array t, or any float—nearest will be used)
# SNAP_T_VALUES = [0.25, 0.50, 0.75]   # edit as desired
SNAP_T_VALUES = [0.56, 0.75, 0.8]   # edit as desired

# Colors for subplot titles
PALETTE = {
    "Truth":            "#000000",
    "QIF":              "#1f6fb4",
    "LIF-32 (conv)":    "#fca082",
    "LIF-32 (cal)":     "#ef6a4b",
    "LIF-64 (conv)":    "#ef3b2c",
    "LIF-64 (cal)":     "#c3271b",
    "LIF-128 (conv)":   "#b2182b",
    "LIF-128 (cal)":    "#7f1010",
}

plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 10.5,
    "savefig.bbox": "tight",
})

# =========================
# Helpers
# =========================
def loadmat(path):
    return {k: np.asarray(v) for k, v in sio.loadmat(path).items() if not k.startswith("__")}

def get_first(d, candidates):
    for k in candidates:
        if k in d:
            return d[k]
    return None

def reorder_to_t_x_y(F, len_t, len_x, len_y):
    """
    Reorder a 3D array to (Nt, Nx, Ny) matching F(t,x,y).
    Tries common permutations; falls back to best-guess by dimension sizes.
    """
    F = np.asarray(F)
    if F.ndim != 3:
        raise ValueError(f"Expected 3D field, got {F.shape}")

    candidate_perms = [
        (2, 0, 1),  # (t,x,y)
        (0, 1, 2),  # already (t,x,y)
        (0, 2, 1),  # (t,y,x)
        (1, 0, 2),  # (x,t,y)
        (2, 1, 0),  # (y,x,t)
        (1, 2, 0),  # (x,y,t)
    ]
    target = (len_t, len_x, len_y)
    for p in candidate_perms:
        cand = F.transpose(p)
        if cand.shape == target:
            return cand

    shape = np.array(F.shape)
    idx_t = np.argmin(np.abs(shape - len_t))
    idx_x = np.argmin(np.abs(shape - len_x))
    idx_y = np.argmin(np.abs(shape - len_y))
    guess = F.transpose(idx_t, idx_x, idx_y)
    if guess.shape != target:
        raise ValueError(f"Cannot reorder {F.shape} to (Nt,Nx,Ny)={target}")
    return guess

def mesh_t_x_y(t, x, y):
    return np.meshgrid(t, x, y, indexing="ij")

def subsample(T, X, Y, F, st=SUB_T, sx=SUB_X, sy=SUB_Y):
    return (T[::st, ::sx, ::sy],
            X[::st, ::sx, ::sy],
            Y[::st, ::sx, ::sy],
            F[::st, ::sx, ::sy])

def nice_3d_axes(ax):
    ax.view_init(elev=22, azim=-55)
    ax.xaxis._axinfo["grid"]['linestyle'] = '--'
    ax.yaxis._axinfo["grid"]['linestyle'] = '--'
    ax.zaxis._axinfo["grid"]['linestyle'] = '--'
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_alpha(0.04)

def scatter_cube(ax, T, X, Y, F, vmin, vmax, title, cmap="viridis"):
    sc = ax.scatter(T, X, Y, c=F, cmap=cmap, marker='o',
                    vmin=vmin, vmax=vmax, edgecolors='none', s=6)
    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_zlabel("y")
    ax.set_title(title, pad=8, color=PALETTE.get(title, "#222"))
    nice_3d_axes(ax)
    return sc

def shared_limits(arrs, vmin_fixed, vmax_fixed):
    if vmin_fixed is not None and vmax_fixed is not None:
        return vmin_fixed, vmax_fixed
    vals = np.concatenate([a.ravel() for a in arrs if a is not None])
    vmin = np.min(vals) if vmin_fixed is None else vmin_fixed
    vmax = np.max(vals) if vmax_fixed is None else vmax_fixed
    if vmin == vmax:
        vmin -= 1e-6; vmax += 1e-6
    return float(vmin), float(vmax)

def metrics_flat(y_true, y_pred):
    yt = y_true.ravel(); yp = y_pred.ravel()
    mae  = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp)**2))
    denom = np.sum((yt - yt.mean())**2) + 1e-12
    r2 = 1.0 - np.sum((yt - yp)**2)/denom
    rel = 100.0*np.linalg.norm(yt-yp)/(np.linalg.norm(yt) + 1e-12)
    return dict(MAE=mae, RMSE=rmse, R2=r2, RelL2_pct=rel)

def nearest_index(arr, val):
    arr = np.asarray(arr).reshape(-1)
    return int(np.argmin(np.abs(arr - val)))

def rect_extent(x, y):
    # extent for imshow with x,y as axes
    return [x.min(), x.max(), y.min(), y.max()]

# =========================
# Plotting
# =========================
def plot_grid_scatter(field_map, t, x, y, title, vmin, vmax, outpath,
                      cmap="viridis", cbar_label=None):
    """
    field_map: dict {name: ndarray (Nt,Nx,Ny)}. May include 'Truth'.
    Creates a grid of 3D scatter “cubes” (t,x,y) with a shared colorbar.
    """
    keys = list(field_map.keys())
    names = (["Truth"] + [k for k in keys if k != "Truth"]) if "Truth" in keys else keys
    n = len(names)

    # fixed to 2x4 (up to 8 panels); adapt as needed
    nrows, ncols = 2, 4
    fig = plt.figure(figsize=(4.2*ncols + 0.8, 3.8*nrows))
    axes = [fig.add_subplot(nrows, ncols, i+1, projection='3d') for i in range(nrows*ncols)]

    last_sc = None
    T, X, Y = mesh_t_x_y(t, x, y)
    for ax, name in zip(axes, names):
        F = field_map[name]
        Ts, Xs, Ys, Fs = subsample(T, X, Y, F)
        sc = scatter_cube(ax, Ts, Xs, Ys, Fs, vmin, vmax, name, cmap=cmap)
        last_sc = sc
    for ax in axes[len(names):]:
        ax.axis("off")

    cbar = fig.colorbar(last_sc, ax=axes)
    if cbar_label is None:
        cbar_label = title.split()[0].lower()
    cbar.set_label(cbar_label, rotation=0, labelpad=8)

    fig.suptitle(title, fontsize=13, y=0.98)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_snapshot_fields(field_map, truth, x, y, t, t_vals, field_name, outdir):
    """
    For each selected time, plot 2D field snapshots for Truth and all models on one figure.
    field_map: dict {name: (Nt,Nx,Ny)}
    truth:     ndarray (Nt,Nx,Ny)
    """
    keys = list(field_map.keys())
    names = (["Truth"] + [k for k in keys if k != "Truth"]) if "Truth" in keys else keys

    # Shared color limits from all models + truth
    all_arrs = [truth] + [field_map[k] for k in keys if k != "Truth"]
    vmin, vmax = shared_limits(all_arrs, None, None)

    for tsel in t_vals:
        tidx = nearest_index(t, tsel)
        tval = float(t[tidx])

        n = len(names)
        # choose grid (<=6): 2x3; (<=8): 2x4; else 3x4
        if n <= 6:
            nrows, ncols = 2, 3
        elif n <= 8:
            nrows, ncols = 2, 4
        else:
            nrows, ncols = 3, 4

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4*ncols, 3.0*nrows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        last_im = None
        for ax, name in zip(axes, names):
            F = truth if name == "Truth" else field_map[name]
            img = F[tidx]  # (Nx,Ny)
            im = ax.imshow(img.T, origin="lower", extent=rect_extent(x, y),
                           cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            last_im = im
            ax.set_title(f"{name}", color=PALETTE.get(name, "#222"), fontsize=11)
            ax.set_xlabel("x"); ax.set_ylabel("y")
        for ax in axes[len(names):]:
            ax.axis("off")

        cbar = fig.colorbar(last_im, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label(field_name, rotation=0, labelpad=8)

        fig.suptitle(f"{field_name} snapshots at t ≈ {tval:.3f}", fontsize=13)
        outpath = Path(outdir) / f"ns_{field_name}_snapshots_t{tidx:03d}.png"
        fig.savefig(outpath, dpi=300)
        plt.close(fig)

def plot_snapshot_errors(field_map, truth, x, y, t, t_vals, field_name, outdir):
    """
    For each selected time, plot 2D absolute error snapshots |Truth - Model|.
    field_map: dict {name: (Nt,Nx,Ny)} including 'Truth' or not (ignored in errors).
    """
    model_names = [k for k in field_map.keys() if k != "Truth"]
    if not model_names:
        return

    for tsel in t_vals:
        tidx = nearest_index(t, tsel)
        tval = float(t[tidx])

        errs = [np.abs(field_map[name][tidx] - truth[tidx]) for name in model_names]
        emax = max(float(e.max()) for e in errs) if errs else 1.0

        n = len(model_names)
        if n <= 6:
            nrows, ncols = 2, 3
        elif n <= 8:
            nrows, ncols = 2, 4
        else:
            nrows, ncols = 3, 4

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4*ncols, 3.0*nrows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        last_im = None
        for ax, name, e in zip(axes, model_names, errs):
            im = ax.imshow(e.T, origin="lower", extent=rect_extent(x, y),
                           cmap="magma", vmin=0.0, vmax=emax, aspect="auto")
            last_im = im
            ax.set_title(f"{name}  (|err|)", color=PALETTE.get(name, "#222"), fontsize=11)
            ax.set_xlabel("x"); ax.set_ylabel("y")
        for ax in axes[len(errs):]:
            ax.axis("off")

        cbar = fig.colorbar(last_im, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label("|err|", rotation=0, labelpad=8)

        fig.suptitle(f"|{field_name} - ŷ| at t ≈ {tval:.3f}", fontsize=13)
        outpath = Path(outdir) / f"ns_{field_name}_errors_t{tidx:03d}.png"
        fig.savefig(outpath, dpi=300)
        plt.close(fig)

# =========================
# Main
# =========================
def main(outdir="."):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    saved = []

    # --- Load truth ---
    dT = loadmat(FILES_TRUTH["mat"])
    x = np.asarray(dT[FILES_TRUTH["x"]]).reshape(-1)
    y = np.asarray(dT[FILES_TRUTH["y"]]).reshape(-1)
    t = np.asarray(dT[FILES_TRUTH["t"]]).reshape(-1)

    U_true = np.asarray(dT[FILES_TRUTH["u"]])
    V_true = np.asarray(dT[FILES_TRUTH["v"]])

    Nt, Nx, Ny = len(t), len(x), len(y)
    U_true = reorder_to_t_x_y(U_true, Nt, Nx, Ny)
    V_true = reorder_to_t_x_y(V_true, Nt, Nx, Ny)

    # --- Collect predictions ---
    U_map = {"Truth": U_true}
    V_map = {"Truth": V_true}

    for name, fpath in FILES_MODELS.items():
        p = Path(fpath)
        if not p.exists():
            print(f"[warn] missing {fpath} -> skip {name}")
            continue
        dm = loadmat(fpath)
        u_pred = get_first(dm, PRED_KEYS["u"])
        v_pred = get_first(dm, PRED_KEYS["v"])
        if u_pred is None and v_pred is None:
            print(f"[warn] {fpath} has no u_pred/v_pred -> skip {name}")
            continue
        if u_pred is not None:
            U_map[name] = reorder_to_t_x_y(u_pred, Nt, Nx, Ny)
        if v_pred is not None:
            V_map[name] = reorder_to_t_x_y(v_pred, Nt, Nx, Ny)

    # --- Metrics (flattened over space-time) ---
    rows = []
    for name, Uhat in U_map.items():
        if name == "Truth": continue
        rows.append({"Model": name, "Field": "u", **metrics_flat(U_true, Uhat)})
    for name, Vhat in V_map.items():
        if name == "Truth": continue
        rows.append({"Model": name, "Field": "v", **metrics_flat(V_true, Vhat)})
    if rows:
        df = pd.DataFrame(rows).set_index(["Field", "Model"]).sort_index()
        csv_path = outdir / "ns_metrics.csv"
        df.to_csv(csv_path)
        print("\n=== Navier–Stokes metrics (flattened) ===")
        with pd.option_context("display.precision", 6):
            print(df)
        print()
        saved.append(csv_path)

    # --- Shared color limits for 3D fields ---
    u_vmin, u_vmax = shared_limits(list(U_map.values()), U_VMIN, U_VMAX)
    v_vmin, v_vmax = shared_limits(list(V_map.values()), V_VMIN, V_VMAX)

    # --- 3D comparison grids ---
    fig_u = outdir / "ns_u_comparison_3d.png"
    plot_grid_scatter(U_map, t, x, y,
                      title="u (x,y,t) — comparison",
                      vmin=u_vmin, vmax=u_vmax,
                      outpath=fig_u,
                      cmap="viridis", cbar_label="u")
    saved.append(fig_u)

    fig_v = outdir / "ns_v_comparison_3d.png"
    plot_grid_scatter(V_map, t, x, y,
                      title="v (x,y,t) — comparison",
                      vmin=v_vmin, vmax=v_vmax,
                      outpath=fig_v,
                      cmap="viridis", cbar_label="v")
    saved.append(fig_v)

    # --- 3D absolute error grids ---
    U_err_map = {name: np.abs(U_map[name] - U_true) for name in U_map if name != "Truth"}
    V_err_map = {name: np.abs(V_map[name] - V_true) for name in V_map if name != "Truth"}

    if U_err_map:
        u_emax = max(float(e.max()) for e in U_err_map.values())
        fig_ue = outdir / "ns_u_error_3d.png"
        plot_grid_scatter(U_err_map, t, x, y,
                          title="|u - ū| (x,y,t) — absolute error",
                          vmin=0.0, vmax=u_emax,
                          outpath=fig_ue,
                          cmap="magma", cbar_label="|err|")
        saved.append(fig_ue)

    if V_err_map:
        v_emax = max(float(e.max()) for e in V_err_map.values())
        fig_ve = outdir / "ns_v_error_3d.png"
        plot_grid_scatter(V_err_map, t, x, y,
                          title="|v - ṽ| (x,y,t) — absolute error",
                          vmin=0.0, vmax=v_emax,
                          outpath=fig_ve,
                          cmap="magma", cbar_label="|err|")
        saved.append(fig_ve)

    # --- Snapshots: solutions (2D) at selected times ---
    plot_snapshot_fields(U_map, U_true, x, y, t, SNAP_T_VALUES, field_name="u", outdir=outdir)
    plot_snapshot_fields(V_map, V_true, x, y, t, SNAP_T_VALUES, field_name="v", outdir=outdir)

    # --- Snapshots: absolute errors (2D) at selected times ---
    plot_snapshot_errors(U_map, U_true, x, y, t, SNAP_T_VALUES, field_name="u", outdir=outdir)
    plot_snapshot_errors(V_map, V_true, x, y, t, SNAP_T_VALUES, field_name="v", outdir=outdir)

    print("Saved files:")
    for p in saved:
        print("  -", Path(p).resolve())

if __name__ == "__main__":
    main(".")
