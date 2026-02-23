# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import interpolate
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap


purple_seq = ["#F65027", "#FD9C44", "#FFF39C", "#d9c8cfff", "#a6999eff"][::-1]
red_seq    = ["#1E3A77", "#4184C3", "#B8E4F8", "#d9c8cfff", "#a6999eff"][::-1]
purple_cmap = LinearSegmentedColormap.from_list("purple_sequential", purple_seq, N=256)
red_cmap    = LinearSegmentedColormap.from_list("red_sequential",    red_seq,    N=256)

FONT_FAMILY = "Arial"
SIZE_TITLE  = 14
SIZE_LABEL  = 12
SIZE_TICK   = 10
SIZE_LEGEND = 10
LINE_W      = 1.7
SPINE_LW    = 1.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
    "axes.edgecolor": "black",
    "axes.linewidth": SPINE_LW,
    "axes.labelsize": SIZE_LABEL,
    "xtick.labelsize": SIZE_TICK,
    "ytick.labelsize": SIZE_TICK,
    "legend.fontsize": SIZE_LEGEND,
    "figure.dpi": 600,
    "xtick.direction": "out",
    "ytick.direction": "out"
})

core_titles = ["CTRL", "TO", "TIO", "TAO", "TPO"]
colors = {
    "CTRL": "#34314c",
    "TO"  : "#47b8e0",
    "TIO" : "#f26d5b",
    "TAO" : "#F4B342",
    "TPO" : "#75B06F",
}

def _as_pct(arr):
    arr = np.asarray(arr, dtype=float)
    return arr * 100.0 if np.nanmax(arr) <= 1.0 else arr

def load_results_from_dir(base_dir: str):
    results = {}
    for t in core_titles:
        p = os.path.join(base_dir, f"{t}_eof_phase_all.nc")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")
        with xr.open_dataset(p) as ds:
            EOF1 = np.asarray(ds["EOF1"])
            EOF2 = np.asarray(ds["EOF2"])
            PC1  = np.asarray(ds["PC1"])
            PC2  = np.asarray(ds["PC2"])
            LEV  = np.asarray(ds["lev"])
            PCT  = _as_pct(np.asarray(ds["percentContrib"]))
            if LEV[0] < LEV[-1]:
                LEV, EOF1, EOF2 = LEV[::-1], EOF1[::-1], EOF2[::-1]
            results[t] = {
                "EOF1": EOF1, "EOF2": EOF2, "PC1": PC1, "PC2": PC2, "lev": LEV,
                "pct1": float(PCT[0]) if PCT.size > 0 else np.nan,
                "pct2": float(PCT[1]) if PCT.size > 1 else np.nan,
            }
    return results

duration_phase = pd.DataFrame({
    "Dataset": ["CTRL", "TO", "TIO", "TAO", "TPO"],
    "Phase_1": [3.81, 4.63, 3.37, 4.11, 5.91],
    "Phase_2": [4.44, 3.95, 4.68, 3.84, 5.82],
    "Phase_3": [3.69, 2.53, 2.68, 1.95, 4.73],
    "Phase_4": [4.12, 3.68, 3.21, 4.21, 5.45],
    "Phase_5": [4.75, 4.00, 3.84, 3.74, 4.45],
    "Phase_6": [3.00, 2.05, 2.53, 2.58, 6.09],
    "Phase_7": [2.38, 3.21, 3.00, 2.11, 2.09],
    "Phase_8": [4.81, 2.89, 3.79, 3.95, 5.55],
})

amplitude_phase = pd.DataFrame({
    "Dataset": ["CTRL", "TO", "TIO", "TAO", "TPO"],
    "Phase_1": [20.4325, 17.6933, 15.7596, 21.9036, 10.436],
    "Phase_2": [18.5846, 14.0315, 14.9601, 19.4941, 9.7356],
    "Phase_3": [16.2413, 13.6006, 13.8366, 16.9588, 9.6131],
    "Phase_4": [18.8559, 16.7278, 16.3320, 22.1273, 8.9008],
    "Phase_5": [20.1011, 17.7347, 17.2929, 21.8331, 9.1789],
    "Phase_6": [17.2760, 13.7701, 14.6647, 17.6357, 10.4331],
    "Phase_7": [16.9412, 16.2617, 14.7458, 21.2759, 8.3098],
    "Phase_8": [23.6969, 20.5006, 19.2201, 25.7161, 12.2564],
})

period_data = {
    'CTRL': [25, 28, 32, 36, 35, 36, 25, 32, 22, 30, 32, 26, 29, 26],
    'TO':   [26, 23, 30, 32, 31, 27, 27, 24, 24, 25, 19, 31, 33, 26, 17, 26],
    'TIO':  [25, 25, 25, 27, 31, 26, 23, 25, 30, 29, 25, 23, 27, 22, 27, 33],
    'TAO':  [26, 26, 26, 28, 21, 24, 24, 23, 27, 28, 26, 30, 24, 28, 25, 29, 20],
    'TPO':  [35, 35, 45, 40, 21, 26, 36, 46, 56, 45]
}
amplitude_data = {
    'CTRL': [19.82052, 20.33609, 20.08066, 21.33975, 17.91511, 18.76228, 20.37227, 20.99469,
             19.70783, 17.53653, 22.22166, 21.9996, 20.81192, 16.94102],
    'TO':   [16.7186, 18.16278, 18.88796, 15.23749, 20.0634, 17.33908, 18.28417, 17.05145,
             18.40622, 17.65401, 14.15963, 20.15361, 18.6846, 19.44674, 16.00195, 18.07549],
    'TIO':  [15.89055, 19.657, 15.53007, 17.89262, 14.93341, 18.38669, 16.64874, 18.09192,
             19.31647, 16.55471, 18.10233, 15.66357, 15.87462, 15.25328, 15.47771, 11.90141],
    'TAO':  [19.2192, 20.66276, 20.26308, 21.95719, 20.73559, 21.52924, 22.25489, 22.37808,
             23.02606, 19.07643, 21.55404, 24.27611, 21.1553, 21.29619, 20.86138, 24.61554, 21.30469],
    'TPO':  [13.39478, 10.73483, 10.72509, 8.922667, 5.745746, 9.034272, 10.94133, 13.98211,
             10.75812, 10.87282]
}

polar_datasets = ["CTRL", "TO", "TIO", "TAO", "TPO"]
phases_cols = [f"Phase_{i}" for i in range(1, 9)]

def build_phase_matrix(phase_df, datasets_order):
    M = np.full((len(datasets_order), 8), np.nan, dtype=float)
    dd = phase_df.set_index("Dataset")
    for i, d in enumerate(datasets_order):
        if d in dd.index:
            M[i, :] = dd.loc[d, phases_cols].values.astype(float)
    return M, float(np.nanmin(M)), float(np.nanmax(M))

def draw_inner_heatmap(ax, M, datasets_order, cmap, vmin, vmax,
                       r_in=0.5, r_out=2.30,
                       theta_start=5*np.pi/6, theta_end=np.pi/6):
    nD = len(datasets_order)
    theta_edges = np.linspace(theta_start, theta_end, nD + 1)
    r_edges = np.linspace(r_in, r_out, 9)
    denom = (vmax - vmin) if (vmax > vmin) else 1.0

    for i in range(nD):
        th1, th2 = theta_edges[i], theta_edges[i+1]
        theta_strip = np.linspace(th1, th2, 40)
        for j in range(8):
            val = M[i, j]
            if np.isnan(val):
                continue
            r1, r2 = r_edges[j], r_edges[j+1]
            xx = np.concatenate([np.cos(theta_strip)*r1, np.cos(theta_strip[::-1])*r2])
            yy = np.concatenate([np.sin(theta_strip)*r1, np.sin(theta_strip[::-1])*r2])
            ax.fill(xx, yy, facecolor=cmap((val - vmin)/denom),
                    edgecolor='white', linewidth=0.8, zorder=1)

    theta_arc = np.linspace(theta_start, theta_end, 200)
    ax.plot(np.cos(theta_arc)*r_out, np.sin(theta_arc)*r_out, 'k-', lw=1.0, zorder=2)
    ax.set_aspect('equal')
    ax.axis('off')
    return theta_edges, r_in, r_out

def draw_outer_violins(ax, dist_dict, datasets_order, dataset_colors,
                       theta_start=5*np.pi/6, theta_end=np.pi/6,
                       r_base=2.55, r_top=3.45, width_scale=0.75,
                       y_range=None):
    nD = len(datasets_order)
    theta_edges   = np.linspace(theta_start, theta_end, nD + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    theta_widths  = np.abs(np.diff(theta_edges))

    if y_range is None:
        all_vals = []
        for v in dist_dict.values():
            vv = np.asarray(v, dtype=float)
            vv = vv[np.isfinite(vv)]
            if vv.size:
                all_vals.append(vv)
        if not all_vals:
            y_min, y_max = 0.0, 1.0
        else:
            all_vals = np.concatenate(all_vals)
            y_min, y_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    else:
        y_min, y_max = map(float, y_range)

    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min, y_max = 0.0, 1.0

    for i, dsname in enumerate(datasets_order):
        vals = np.asarray(dist_dict.get(dsname, []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue

        kde  = gaussian_kde(vals)
        yi   = np.linspace(y_min, y_max, 160)
        dens = kde(yi)
        if dens.max() > 0:
            dens = dens / dens.max()

        rr    = r_base + (yi - y_min) / (y_max - y_min) * (r_top - r_base)
        halfw = dens * (theta_widths[i] * rr) * width_scale * 0.5

        x_local = np.concatenate([rr[::-1], rr])
        y_local = np.concatenate([-halfw[::-1], halfw])

        th = theta_centers[i]
        ct, st = np.cos(th), np.sin(th)
        xg = ct * x_local - st * y_local
        yg = st * x_local + ct * y_local

        ax.fill(
            xg, yg,
            facecolor=dataset_colors.get(dsname, "#999999"),
            edgecolor='black', linewidth=0.5, alpha=0.70, zorder=4
        )

        for stat, lw in zip(np.percentile(vals, [25, 50, 75]), [0.8, 1.2, 0.8]):
            stat = float(np.clip(stat, y_min, y_max))
            r_stat = r_base + (stat - y_min) / (y_max - y_min) * (r_top - r_base)
            hw = 0.22 * theta_widths[i] * r_stat * width_scale
            x_line = np.array([r_stat, r_stat])
            y_line = np.array([-hw, hw])
            ax.plot(ct*x_line - st*y_line, st*x_line + ct*y_line, color='k', lw=lw, zorder=5)

        r_lab = r_top + 0.06
        ax.text(np.cos(th)*r_lab, np.sin(th)*r_lab, dsname,
                ha='center', va='bottom', rotation=np.degrees(th)-90,
                fontsize=SIZE_TICK, fontweight='bold', zorder=6)

    theta_arc = np.linspace(theta_start, theta_end, 300)
    ax.plot(np.cos(theta_arc)*r_top, np.sin(theta_arc)*r_top, color="black", lw=1.2, zorder=6)
    return y_min, y_max, r_base, r_top

def create_polar_panel(ax, phase_table, dist_dict, title, cmap,
                       inner_r_out=2.30, vio_r_base=2.55, vio_r_top=3.45,
                       y_range=None, ring_ticks=None,
                       inner_vmin=None, inner_vmax=None,
                       violin_ylabel=None):
    M, vmin, vmax = build_phase_matrix(phase_table, polar_datasets)
    if inner_vmin is not None:
        vmin = float(inner_vmin)
    if inner_vmax is not None:
        vmax = float(inner_vmax)

    _, _, _ = draw_inner_heatmap(ax, M, polar_datasets, cmap, vmin, vmax,
                                 r_in=0.5, r_out=inner_r_out)

    y_min, y_max, r_base, r_top = draw_outer_violins(
        ax, dist_dict, polar_datasets, colors,
        r_base=vio_r_base, r_top=vio_r_top, y_range=y_range
    )

    if ring_ticks is not None:
        theta_arc = np.linspace(5*np.pi/6, np.pi/6, 300)
        for v in ring_ticks:
            v = float(v)
            rr = r_base + (v - y_min)/(y_max - y_min) * (r_top - r_base)
            ax.plot(np.cos(theta_arc)*rr, np.sin(theta_arc)*rr, color="0.75", lw=0.8, ls="--", zorder=3)

    if violin_ylabel:
        ax.text(-r_top*0.98, r_top*0.55, violin_ylabel, fontsize=SIZE_LABEL,
                rotation=90, ha='center', va='center')

    R = r_top + 0.22
    ax.set_xlim(-R, R)
    ax.set_ylim(0, R)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=SIZE_TITLE, pad=12)
    return vmin, vmax

def plot_eof_panels(results):
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(4.2, 1.5), sharey=True, gridspec_kw={'wspace': 0.15}
    )

    for t in core_titles:
        r = results[t]
        ax_a.plot(r["EOF1"], r["lev"], '-', color=colors[t], lw=LINE_W)

    ax_a.axvline(0, color='black', linestyle=':', linewidth=1.1, alpha=0.8)
    ax_a.set_ylim(float(np.nanmax(results["CTRL"]["lev"])), float(np.nanmin(results["CTRL"]["lev"])))
    ax_a.set_xlim(-1, 1)
    ax_a.set_xticks([-1, 0, 1])
    ax_a.set_yticks([70, 50, 30, 20, 10])
    ax_a.set_title("(a) EOF1", fontsize=SIZE_TITLE, loc='left')
    ax_a.set_xlabel("Correlation")
    ax_a.set_ylabel("Pressure (hPa)")
    ax_a.tick_params(axis='both', labelsize=SIZE_TICK)

    labels_a = [f"{t}: {int(round(results[t]['pct1']))}%" for t in core_titles]
    handles_a = [plt.Line2D([], [], color="none") for _ in core_titles]
    leg_a = ax_a.legend(handles_a, labels_a, loc='lower right', frameon=False, fontsize=8,
                       handlelength=0.0, handletextpad=0.0, labelspacing=0.12)
    for text, t in zip(leg_a.get_texts(), core_titles):
        text.set_color(colors[t])

    for t in core_titles:
        r = results[t]
        ax_b.plot(r["EOF2"], r["lev"], '-', color=colors[t], lw=LINE_W)

    ax_b.axvline(0, color='black', linestyle=':', linewidth=1.1, alpha=0.8)
    ax_b.set_xlim(-1, 1)
    ax_b.set_xticks([-1, 0, 1])
    ax_b.set_yticks([70, 50, 30, 20, 10])
    ax_b.set_title("(b) EOF2", fontsize=SIZE_TITLE, loc='left')
    ax_b.set_xlabel("Correlation")
    ax_b.tick_params(axis='both', labelsize=SIZE_TICK)

    labels_b = [f"{t}: {int(round(results[t]['pct2']))}%" for t in core_titles]
    handles_b = [plt.Line2D([], [], color="none") for _ in core_titles]
    leg_b = ax_b.legend(handles_b, labels_b, loc='lower right', frameon=False, fontsize=8,
                       handlelength=0.0, handletextpad=0.0, labelspacing=0.12)
    for text, t in zip(leg_b.get_texts(), core_titles):
        text.set_color(colors[t])

    for ax in (ax_a, ax_b):
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(SPINE_LW)

    plt.tight_layout()
    plt.show()

def plot_phase_space(results):
    fig = plt.figure(figsize=(4.2, 4.0))
    ax = fig.add_subplot(111)

    all_p1 = np.concatenate([np.asarray(results[t]["PC1"], dtype=float) for t in core_titles])
    all_p2 = np.concatenate([np.asarray(results[t]["PC2"], dtype=float) for t in core_titles])
    all_p1 = all_p1[np.isfinite(all_p1)]
    all_p2 = all_p2[np.isfinite(all_p2)]
    r = np.hypot(all_p1, all_p2)
    rmax = float(np.nanpercentile(r, 99.5)) if r.size else 1.0
    lim = max(1.0, rmax) * 1.05

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])

    ref_style = dict(color='lightgray', linestyle=':', linewidth=1.0, alpha=0.8, zorder=0)
    ax.axhline(0, **ref_style)
    ax.axvline(0, **ref_style)
    ax.plot([-lim, lim], [-lim, lim], **ref_style)
    ax.plot([-lim, lim], [lim, -lim], **ref_style)

    for t in core_titles:
        p1 = np.asarray(results[t]["PC1"], dtype=float)
        p2 = np.asarray(results[t]["PC2"], dtype=float)
        ok = np.isfinite(p1) & np.isfinite(p2)
        p1, p2 = p1[ok], p2[ok]
        ax.scatter(p1, p2, c=[colors[t]], alpha=0.15, s=5, edgecolors='none')

        rads = np.hypot(p1, p2)
        thetas = np.arctan2(p2, p1)
        bins = np.linspace(-np.pi, np.pi, 9)

        cen, rad = [], []
        for j in range(8):
            sel = (thetas >= bins[j]) & (thetas < bins[j+1])
            if np.any(sel):
                cen.append((bins[j] + bins[j+1]) / 2)
                rad.append(float(np.nanmean(rads[sel])))

        if len(cen) >= 3:
            cen = np.array(cen, dtype=float)
            rad = np.array(rad, dtype=float)
            cen = np.append(cen, cen[0] + 2*np.pi)
            rad = np.append(rad, rad[0])
            angs = np.linspace(cen[0], cen[-1], 300)
            try:
                tck = interpolate.splrep(cen, rad, s=0, per=True)
                r_s = interpolate.splev(angs, tck)
            except Exception:
                r_s, angs = rad, cen
            ax.plot(np.asarray(r_s)*np.cos(angs), np.asarray(r_s)*np.sin(angs),
                    lw=LINE_W, color=colors[t], label=t)

    phase_centers_deg = np.arange(22.5, 360, 45)
    phase_labels = [f"P{i}" for i in range(1, 9)]
    r_text = lim * 0.93
    for th_deg, label in zip(phase_centers_deg, phase_labels):
        th = np.deg2rad(th_deg)
        ax.text(r_text*np.cos(th), r_text*np.sin(th), label,
                ha='center', va='center', fontsize=SIZE_TICK, color='black', zorder=3)

    ax.set_title("(c) Phase space", fontsize=SIZE_TITLE, loc='left')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.tick_params(axis='both', labelsize=SIZE_TICK)
    ax.set_axisbelow(True)

    leg = ax.legend(loc='upper right', frameon=False, fontsize=8,
                    labelspacing=0.2, handletextpad=0.3, borderaxespad=0.2)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())

    plt.tight_layout()
    plt.show()

def plot_polar_panels():
    fig = plt.figure(figsize=(5, 4.8))
    ax = fig.add_subplot(111)
    vmin_amp, vmax_amp = create_polar_panel(
        ax, amplitude_phase, amplitude_data, "(d) Amplitude",
        cmap=purple_cmap, y_range=None, ring_ticks=None,
        inner_vmin=None, inner_vmax=None,
        violin_ylabel=r"m s$^{-1}$"
    )
    sm = plt.cm.ScalarMappable(cmap=purple_cmap, norm=Normalize(vmin=vmin_amp, vmax=vmax_amp))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.035, pad=0.02)
    cbar.set_label('Amplitude (m s$^{-1}$)', fontsize=SIZE_LABEL)
    cbar.ax.tick_params(labelsize=SIZE_TICK)
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(5, 4.8))
    ax = fig.add_subplot(111)
    vmin_per, vmax_per = create_polar_panel(
        ax, duration_phase, period_data, "(e) Period",
        cmap=red_cmap, y_range=None, ring_ticks=None,
        inner_vmin=None, inner_vmax=None,
        violin_ylabel="months"
    )
    sm = plt.cm.ScalarMappable(cmap=red_cmap, norm=Normalize(vmin=vmin_per, vmax=vmax_per))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.035, pad=0.02)
    cbar.set_label('Phase duration (months)', fontsize=SIZE_LABEL)
    cbar.ax.tick_params(labelsize=SIZE_TICK)
    plt.tight_layout()
    plt.show()

def main():
    base_dir = os.getenv("EOF_PHASE_DIR", ".")
    results = load_results_from_dir(base_dir)
    plot_eof_panels(results)
    plot_phase_space(results)
    plot_polar_panels()

if __name__ == "__main__":
    main()
