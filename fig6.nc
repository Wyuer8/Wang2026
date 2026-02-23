# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmaps

warnings.filterwarnings("ignore")


EXPS = ["CTRL", "TO", "TIO", "TAO", "TPO"]
VAR_ORDER = ["divF", "vadv", "butgwspec"]
VAR_MULT  = {"butgwspec": 86400.0, "vadv": 1.0, "divF": 1.0}

VAR_TITLES = {
    "divF":      "resolved wave forcing",
    "vadv":      "vertical advection",
    "butgwspec": "GW forcing",
}

PHASE_LABELS = ["p1","p2","p3","p4","p5","p6","p7","p8","p1"]
div_cmap = cmaps.cmp_b2r


def _find_name(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    return None


def _lat_band_mean(da, lat_name, lat_min=-5.0, lat_max=5.0):
    if (lat_name is None) or (lat_name not in da.dims):
        return da
    da2 = da.sel({lat_name: slice(lat_min, lat_max)})
    if da2.sizes.get(lat_name, 0) == 0:
        da2 = da.sel({lat_name: slice(lat_max, lat_min)})
    if da2.sizes.get(lat_name, 0) == 0:
        return da
    return da2.mean(lat_name, keep_attrs=True)


def load_tem(path):
    ds = xr.open_dataset(path)
    lat = _find_name(ds, ["lat", "latitude", "y"])
    lev = _find_name(ds, ["lev", "level", "plev", "p"])
    pha = _find_name(ds, ["phase", "Phase", "qbo_phase", "phi"])

    keep = {}
    for v in ["divF", "vadv", "wstar", "um"]:
        if v in ds:
            da = _lat_band_mean(ds[v], lat)
            for d in list(da.dims):
                if d not in ("phase", "lev"):
                    da = da.mean(d, keep_attrs=True)
            da = da.squeeze()
            if "phase" in da.dims and "lev" in da.dims:
                da = da.transpose("phase", "lev")
            keep[v] = da

    keep[pha] = ds[pha]
    keep[lev] = ds[lev]
    out = xr.Dataset(keep).rename({pha: "phase", lev: "lev"})
    ds.close()
    return out


def load_gw(path):
    ds = xr.open_dataset(path)
    lat = _find_name(ds, ["lat", "latitude", "y"])
    lev = _find_name(ds, ["lev", "level", "plev", "p"])
    pha = _find_name(ds, ["phase", "Phase", "qbo_phase", "phi"])

    keep = {}
    gw_var = "butgwspec" if "butgwspec" in ds else ("utgwspec" if "utgwspec" in ds else None)
    if gw_var is not None:
        da = _lat_band_mean(ds[gw_var], lat)
        if "lon" in da.dims:
            da = da.mean("lon", keep_attrs=True)
        for d in list(da.dims):
            if d not in ("phase", "lev"):
                da = da.mean(d, keep_attrs=True)
        da = da.squeeze()
        if "phase" in da.dims and "lev" in da.dims:
            da = da.transpose("phase", "lev")
        keep["butgwspec"] = da

    keep[pha] = ds[pha]
    keep[lev] = ds[lev]
    out = xr.Dataset(keep).rename({pha: "phase", lev: "lev"})
    ds.close()
    return out


def build_paths(base_dir):
    file_map_tem = {
        "CTRL": os.path.join(base_dir, "CTRL_tem_phase_averaged.nc"),
        "TO":   os.path.join(base_dir, "TO_fixed_tem_phase_averaged.nc"),
        "TIO":  os.path.join(base_dir, "TIO_tem_phase_averaged.nc"),
        "TAO":  os.path.join(base_dir, "TAO_tem_phase_averaged.nc"),
        "TPO":  os.path.join(base_dir, "TPO_fixed_tem_phase_averaged.nc"),
    }
    file_map_gw = {
        "CTRL": os.path.join(base_dir, "CTRL_phase_averaged.nc"),
        "TO":   os.path.join(base_dir, "TO_phase_averaged.nc"),
        "TIO":  os.path.join(base_dir, "TIO_phase_averaged.nc"),
        "TAO":  os.path.join(base_dir, "TAO_phase_averaged.nc"),
        "TPO":  os.path.join(base_dir, "TPO_phase_averaged.nc"),
    }
    return file_map_tem, file_map_gw


def get_Z(ds, varname):
    if varname not in ds:
        return None
    lev = np.asarray(ds["lev"].values, dtype=float)
    x = np.arange(1, 10, 1)

    Z = np.asarray(ds[varname].values, dtype=float).T * VAR_MULT.get(varname, 1.0)

    if Z.shape[1] != 9:
        if Z.shape[1] > 9:
            Z = Z[:, :9]
        else:
            Z = np.pad(Z, ((0, 0), (0, 9 - Z.shape[1])), constant_values=np.nan)

    U = None
    if "um" in ds and set(["phase", "lev"]).issubset(set(ds["um"].dims)):
        U0 = np.asarray(ds["um"].values, dtype=float).T
        if U0.shape[1] != 9:
            if U0.shape[1] > 9:
                U0 = U0[:, :9]
            else:
                U0 = np.pad(U0, ((0, 0), (0, 9 - U0.shape[1])), constant_values=np.nan)
        U = U0

    return x, lev, Z, U


def plot_panel(ax, exp, varname, ds, levels, show_ylabel=False, show_xlabel=False):
    data = get_Z(ds, varname)
    if data is None:
        ax.axis("off")
        return None

    x, lev, Z, U = data

    cf = ax.contourf(x, lev, Z, levels=levels, cmap=div_cmap, extend="both")

    if U is not None:
        ax.contour(x, lev, U, levels=np.arange(-30, 35, 5), colors="k", linewidths=0.6)
        ax.contour(x, lev, U, levels=[0], colors="k", linewidths=1.6)

    ax.invert_yaxis()
    ax.set_yscale("symlog")

    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_LABELS)

    if show_xlabel:
        ax.set_xlabel("Phase")
    if show_ylabel:
        ax.set_ylabel("Pressure (hPa)")

    ax.text(4, 30, "W", fontweight="bold", ha="center", va="center")
    ax.text(7, 30, "E", fontweight="bold", ha="center", va="center")

    ax.set_title(f"{exp} {VAR_TITLES.get(varname, varname)}", loc="left")
    return cf


def main():
    base_dir = os.getenv("QBO_PHASE_DIR", ".")
    out_dir  = os.getenv("FIG_OUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    file_map_tem, file_map_gw = build_paths(base_dir)

    datasets = {}
    for exp in EXPS:
        if not os.path.exists(file_map_tem[exp]):
            raise FileNotFoundError(f"Missing: {file_map_tem[exp]}")
        if not os.path.exists(file_map_gw[exp]):
            raise FileNotFoundError(f"Missing: {file_map_gw[exp]}")
        ds_tem = load_tem(file_map_tem[exp])
        ds_gw  = load_gw(file_map_gw[exp])
        datasets[exp] = xr.merge([ds_tem, ds_gw], compat="override")

    allZ = []
    for exp in EXPS:
        for varname in VAR_ORDER:
            data = get_Z(datasets[exp], varname)
            if data is None:
                continue
            _, _, Z, _ = data
            allZ.append(Z[np.isfinite(Z)])
    allZ = np.concatenate(allZ) if allZ else np.array([0.0, 1.0], dtype=float)

    v = float(np.nanmax(np.abs(allZ))) if np.isfinite(allZ).any() else 1.0
    v = max(v, 1e-12)
    levels = np.linspace(-v, v, 17)

    fig, axs = plt.subplots(
        nrows=3, ncols=5, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True
    )

    mappable = None
    for r, varname in enumerate(VAR_ORDER):
        for c, exp in enumerate(EXPS):
            ax = axs[r, c]
            cf = plot_panel(
                ax, exp, varname, datasets[exp],
                levels=levels,
                show_ylabel=(c == 0),
                show_xlabel=(r == 2),
            )
            if mappable is None and cf is not None:
                mappable = cf

    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axs, orientation="vertical", shrink=0.85, pad=0.02, extend="both")
        cb.set_label(r"$\mathrm{m\ s^{-1}\ day^{-1}}$")

    out_path = os.path.join(out_dir, "Figure6.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to: {out_path}")

    plt.show()

    for ds in datasets.values():
        try:
            ds.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
