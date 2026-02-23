# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

DATA_DIR = os.getenv("DATA_DIR", ".")
FIG_OUT_DIR = os.getenv("FIG_OUT_DIR", ".")
os.makedirs(FIG_OUT_DIR, exist_ok=True)

u_paths = {
    "TAO": os.path.join(DATA_DIR, "walker", "TAO_u_2d.nc"),
    "TPO": os.path.join(DATA_DIR, "walker", "TPO_u_2d.nc"),
}
psi_paths = {
    "TAO": os.path.join(DATA_DIR, "walker", "TAO_walker_2d.nc"),
    "TPO": os.path.join(DATA_DIR, "walker", "TPO_walker_2d.nc"),
}
prect_paths = {
    "CTRL": os.path.join(DATA_DIR, "EXP", "CTRL2000", "PRECT.CTRL2000.nc"),
    "TO":   os.path.join(DATA_DIR, "EXP", "TO2000",   "PRECT.TO2000.nc"),
    "TIO":  os.path.join(DATA_DIR, "EXP", "TIO2000",  "PRECT.TIO2000.nc"),
    "TAO":  os.path.join(DATA_DIR, "EXP", "TAO2000",  "PRECT.TAO2000.nc"),
    "TPO":  os.path.join(DATA_DIR, "EXP", "TPO2000",  "PRECT.TPO2000.nc"),
}

colors = {
    "CTRL": "#34314c",
    "TO":   "#47b8e0",
    "TIO":  "#f26d5b",
    "TAO":  "#FBA518",
    "TPO":  "#6CA651",
}

PRECT = {}
lat2 = None
lon2 = None
for name, p in prect_paths.items():
    ds = xr.open_dataset(p)
    da = ds["PRECT"].mean("time") * 86400.0 * 1000.0
    PRECT[name] = da
    if lat2 is None:
        lat2 = da["lat"].values
        lon2 = da["lon"].values
    ds.close()

PSI = {}
U = {}
lon1 = None

for name in ["TAO", "TPO"]:
    ds_u = xr.open_dataset(u_paths[name])
    u = ds_u["u_eq"]
    lev_u = np.asarray(u["lev"].values, dtype=float)
    if np.nanmax(lev_u) > 2000:
        lev_u = lev_u * 0.01
    u = u.assign_coords(lev=lev_u).sel(lev=slice(1000, 7))
    U[name] = u
    ds_u.close()

    ds_p = xr.open_dataset(psi_paths[name])
    psi = ds_p["psi_walker_mean"]
    lev_p = np.asarray(psi["lev"].values, dtype=float)
    if np.nanmax(lev_p) > 2000:
        lev_p = lev_p * 0.01
    psi = psi.assign_coords(lev=lev_p).sel(lev=slice(1000, 7))
    PSI[name] = psi
    if lon1 is None:
        lon1 = psi["lon"].values
    ds_p.close()

prect_means = {k: v.mean("lon") for k, v in PRECT.items()}

prect_stack = np.concatenate([PRECT["TAO"].values.ravel(), PRECT["TPO"].values.ravel()])
prect_stack = prect_stack[np.isfinite(prect_stack)]
prect_v = np.nanpercentile(prect_stack, 99) if prect_stack.size else 1.0
if not np.isfinite(prect_v) or prect_v == 0:
    prect_v = 1.0
levels_prect = np.linspace(0, prect_v, 11)

psi_stack = np.concatenate([PSI["TAO"].values.ravel(), PSI["TPO"].values.ravel()])
psi_stack = psi_stack[np.isfinite(psi_stack)]
psi_v = np.nanpercentile(np.abs(psi_stack), 99) if psi_stack.size else 1.0
if not np.isfinite(psi_v) or psi_v == 0:
    psi_v = 1.0
levels_psi = np.linspace(-psi_v, psi_v, 17)

levels_u = np.arange(-30, 35, 5)

proj = ccrs.PlateCarree(central_longitude=180)
data_crs = ccrs.PlateCarree()

fig = plt.figure(figsize=(16, 4))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.06, 0.06])

ax_line = fig.add_subplot(gs[0, 0])
for name in ["CTRL", "TO", "TIO", "TAO", "TPO"]:
    ax_line.plot(lat2, prect_means[name].values, label=name, color=colors[name])
ax_line.set_xlabel("Latitude")
ax_line.set_ylabel("Precipitation (mm day$^{-1}$)")
ax_line.set_title("Precipitation (zonal mean)")
ax_line.legend(frameon=False)

ax_map_tao = fig.add_subplot(gs[0, 1], projection=proj)
ax_map_tpo = fig.add_subplot(gs[0, 2], projection=proj)

cf_map = None
for ax, name in [(ax_map_tao, "TAO"), (ax_map_tpo, "TPO")]:
    ax.set_extent([-180, 180, -40, 40], crs=data_crs)
    d_cyc, lon_cyc = add_cyclic_point(np.asarray(PRECT[name].values), coord=np.asarray(lon2))
    cf = ax.contourf(lon_cyc, lat2, d_cyc, levels=levels_prect, transform=data_crs, extend="max")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_title(name)
    if cf_map is None:
        cf_map = cf

cax_prect = fig.add_subplot(gs[1, 1:3])
cb1 = fig.colorbar(cf_map, cax=cax_prect, orientation="horizontal")
cb1.set_label("Precipitation (mm day$^{-1}$)")

cax_psi = fig.add_subplot(gs[2, 1:3])
sm_psi = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=float(levels_psi.min()), vmax=float(levels_psi.max())), cmap=plt.get_cmap())
sm_psi.set_array([])
cb2 = fig.colorbar(sm_psi, cax=cax_psi, orientation="horizontal")
cb2.set_label("Walker streamfunction ψ (auto scale)")

fig.tight_layout()
fig.savefig(os.path.join(FIG_OUT_DIR, "walker_summary.pdf"), bbox_inches="tight")

fig2, axs = plt.subplots(1, 2, figsize=(16, 4), sharey=True)

cf_psi_ref = None
for j, name in enumerate(["TAO", "TPO"]):
    ax = axs[j]
    psi = PSI[name]
    u = U[name]

    psi_cyc, lon_cyc = add_cyclic_point(np.asarray(psi.values), coord=np.asarray(lon1))
    u_cyc, _ = add_cyclic_point(np.asarray(u.values), coord=np.asarray(u["lon"].values if "lon" in u.coords else lon1))

    cf = ax.contourf(lon_cyc, psi["lev"].values, psi_cyc, levels=levels_psi, extend="both")
    ax.contour(lon_cyc, u["lev"].values, u_cyc, levels=levels_u, colors="k", linewidths=0.7)

    ax.set_yscale("symlog")
    ax.invert_yaxis()
    ax.set_xlabel("Longitude")
    ax.set_title(name)
    ax.set_ylabel("Pressure (hPa)")

    if cf_psi_ref is None:
        cf_psi_ref = cf

cb = fig2.colorbar(cf_psi_ref, ax=axs, orientation="vertical", fraction=0.03, pad=0.02)
cb.set_label("Walker streamfunction ψ")

fig2.tight_layout()
fig2.savefig(os.path.join(FIG_OUT_DIR, "walker_psi_u.pdf"), bbox_inches="tight")

plt.show()
