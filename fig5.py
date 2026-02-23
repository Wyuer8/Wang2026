# -*- coding: utf-8 -*-

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


colors = {
    "CTRL": "#34314c",
    "TO"  : "#47b8e0",
    "TIO" : "#f26d5b",
    "TAO" : "#FBA518",
    "TPO" : "#6CA651",
}

core_titles = ["CTRL", "TO", "TIO", "TAO", "TPO"]

def phase_labels_from_radians(phase_rad):
    ang = (np.asarray(phase_rad, dtype=float) + 2*np.pi) % (2*np.pi)
    deg = np.degrees(ang)
    return (np.floor(deg / 45.0).astype(int) % 8) + 1

def get_phase_profiles(ds):
    u   = ds["um_with_phase"]      # (time, lev)
    lev = np.asarray(ds["lev"].values, dtype=float)
    ph  = np.asarray(ds["phase"].values, dtype=float)

    labels = phase_labels_from_radians(ph)  # 1..8

    profiles = np.full((8, lev.size), np.nan, dtype=float)
    for p in range(1, 9):
        sel = (labels == p)
        if np.any(sel):
            profiles[p-1, :] = u.isel(time=sel).mean("time").values
    return profiles, lev

def main():
    data_dir = os.getenv("EOF_PHASE_DIR", ".")
    out_dir  = os.getenv("FIG_OUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    files = {name: os.path.join(data_dir, f"{name}_eof_phase.nc") for name in core_titles}

    fig, axs = plt.subplots(2, 4, figsize=(8, 4))
    axs = axs.flatten()
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    global_xmin, global_xmax = np.inf, -np.inf
    lev_ref = None

    cache = {}
    for name, fp in files.items():
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing file: {fp}")
        with xr.open_dataset(fp) as ds:
            profs, lev = get_phase_profiles(ds)
        cache[name] = (profs, lev)
        lev_ref = lev if lev_ref is None else lev_ref
        global_xmin = min(global_xmin, np.nanmin(profs))
        global_xmax = max(global_xmax, np.nanmax(profs))

    if not np.isfinite(global_xmin) or not np.isfinite(global_xmax):
        raise ValueError("All profiles are NaN; check inputs.")

    pad = 0.05 * (global_xmax - global_xmin if global_xmax > global_xmin else 1.0)
    xlim = (global_xmin - pad, global_xmax + pad)

    for phase_idx in range(8):
        ax = axs[phase_idx]

        for name in core_titles:
            profs, lev = cache[name]
            ax.plot(profs[phase_idx, :], lev, color=colors[name], lw=2)

        ax.invert_yaxis()
        ax.set_xlim(xlim)
        ax.set_title(f"{letters[phase_idx]} Phase {phase_idx + 1}", loc="left")

        ax.set_xlabel("u (m s$^{-1}$)")
        if phase_idx % 4 == 0:
            ax.set_ylabel("Pressure (hPa)")

        ax.axvline(0, color="0.6", linestyle=":", lw=1)

    legend_handles = [Line2D([0, 1], [0, 0], color=colors[name], lw=2) for name in core_titles]
    fig.legend(legend_handles, core_titles, loc="lower center", ncol=len(core_titles),
               frameon=False)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "Figure5.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")

    print(f"Figure saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
