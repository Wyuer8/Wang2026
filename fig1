import os
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter, FuncFormatter
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib import transforms as mtransforms

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# =====================================================
# 0) 路径
# =====================================================
DATA_DIR = r"F:\paper_use\plot_data"
TREND_NC  = os.path.join(DATA_DIR, "obs_trend_mme.nc")
PKL       = os.path.join(DATA_DIR, "timeseries_data.pkl")
CACHE_NC  = os.path.join(DATA_DIR, "qbo_amp_profiles_cache_v4.nc")


# =====================================================
# 0.5) 布局控制
# =====================================================
MOVE_FG_Y_TO_RIGHT = True

BCDE_SHRINK_HEIGHT = 0.040
SHIFT_BC_DOWN = 0.04          # 你原来的
SHIFT_DE_UP   = 0.0           # 你原来的（第三行不动）

# ✅ 本次唯一新增：只下移第二行(b/c)，第三行(d/e)完全不动
BC_ONLY_EXTRA_DOWN = 0.016   # 想“更一点点”就 0.006；想更下就 0.012

HSPACE_MAIN = 0.55

FG_HSPACE = 1.25
GAP_FRAC_RIGHT = 0.17

TITLE_X = -0.05
TITLE_PAD = 6
X_LABELPAD = 0.5

A_RIGHT_TEXT_DY_PT = -2.0
A_BASIN_LABEL_DX_DEG = 1.5
A_BASIN_LABEL_DY_DEG = 1.2

# colorbar 位置（保持你当前版本，不动）
CB_Y_OFFSET = -0.10


# =====================================================
# 1) 字号
# =====================================================
TITLE_FS = 11
XYLABEL_FS = 8
TICK_FS = 8
LEGEND_FS = 8.5


# =====================================================
# 2) 样式
# =====================================================
rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "axes.unicode_minus": False,
    "hatch.linewidth": 0.35,
})

LW_ALL = 2.2

COL_OBS = "#4A4A4A"
COL_HIS = "#63B9FE"
COL_245 = "#FE931A"
COL_585 = "#FA2F42"
ALPHA_OBS = 0.28
ALPHA_HIS = 0.18
ALPHA_245 = 0.18
ALPHA_585 = 0.14

A_SPLIT_COL = "#4A4A4A"
REF_LW_A_BOLD = 1.8

LABEL_COL_A = A_SPLIT_COL
LABEL_FS_A  = 9.2
LABEL_WEIGHT_A = "bold"
LABEL_BBOX_A = dict(
    boxstyle="round,pad=0.22,rounding_size=0.15",
    fc="white", ec=LABEL_COL_A, lw=0.6, alpha=0.92
)

REF_COL = "#B0B0B0"
REF_LW  = 1.0

COL_FUB_F  = "#202020"
COL_ERA5_F = "#7A7A7A"
COL_ERA5   = COL_ERA5_F

LEG_HANDLELENGTH = 1.25
LEG_BORDERPAD = 0.25
LEG_LABELSPACING = 0.22

HEIGHT_RATIOS = [1.00, 1.00, 1.00]


# =====================================================
# 3) 工具函数
# =====================================================
def close_box_axes(ax):
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(top=True, right=True)

def draw_band(ax, med, lo, hi, color, alpha, lw, label,
              z_fill=2.0, z_line=3.0, hatch=None, hatch_alpha=0.85):
    x = med["year"].values
    ax.fill_between(x, lo.values, hi.values,
                    facecolor=color, alpha=alpha,
                    linewidth=0, edgecolor="none",
                    zorder=z_fill)
    if hatch is not None:
        ax.fill_between(x, lo.values, hi.values,
                        facecolor="none", hatch=hatch,
                        edgecolor=color, linewidth=0.0,
                        alpha=hatch_alpha, zorder=z_fill + 0.05)
    ax.plot(x, med.values, color=color, lw=lw, label=label, zorder=z_line)

def add_ordered_legend_be(ax):
    want = ["Observation-based", "Historical", "SSP2-4.5", "SSP5-8.5"]
    handles, labels = ax.get_legend_handles_labels()
    mapping = {lab: h for h, lab in zip(handles, labels)}
    h2, l2 = [], []
    for w in want:
        if w in mapping:
            h2.append(mapping[w])
            l2.append(w)
    ax.legend(
        h2, l2, loc="upper left", frameon=False, fontsize=7.0,
        handlelength=LEG_HANDLELENGTH, borderaxespad=LEG_BORDERPAD,
        labelspacing=LEG_LABELSPACING
    )

def force_align_right_column(ax1, ax4, ax5, ax6, ax7, gap_frac=0.14):
    fig = ax1.figure
    fig.canvas.draw()

    p1 = ax1.get_position()
    p4 = ax4.get_position()
    p5 = ax5.get_position()
    p6 = ax6.get_position()

    top = p1.y1
    bottom = min(p4.y0, p5.y0)

    x0 = p6.x0
    x1 = p6.x1
    w = x1 - x0

    total_h = top - bottom
    gap = total_h * gap_frac
    h = (total_h - gap) / 2.0

    ax7.set_position([x0, bottom, w, h])
    ax6.set_position([x0, bottom + h + gap, w, h])

def set_pressure_axis_integer_ticks(ax, ticks=(70, 50, 30, 20, 10), ylim_top=75, ylim_bottom=9):
    ticks = list(ticks)
    tick_set = set(ticks)
    ax.set_yscale("log")
    ax.set_ylim(ylim_top, ylim_bottom)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(y)}" if y in tick_set else ""))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())

def lon_to_0_360(lon):
    lon = lon % 360
    if lon < 0:
        lon += 360
    return lon

def keep_left_and_add_right_y(ax):
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(axis="y", which="both",
                   left=True, right=True,
                   labelleft=True, labelright=True)
    ax.yaxis.set_label_position("right")

def shrink_axes_vertically(ax_list, shrink_abs=0.05):
    for ax in ax_list:
        p = ax.get_position()
        new_h = max(0.001, p.height - shrink_abs)
        y0 = p.y0 + (p.height - new_h) / 2.0
        ax.set_position([p.x0, y0, p.width, new_h])

def shift_axes_y(ax_list, dy):
    for ax in ax_list:
        p = ax.get_position()
        ax.set_position([p.x0, p.y0 + dy, p.width, p.height])

def add_title_right_text_aligned_to_title(ax, text, fontsize, color, pad, dy_pt=0.0):
    fig = ax.figure
    off = mtransforms.ScaledTranslation(0, (pad + dy_pt) / 72.0, fig.dpi_scale_trans)
    ax.text(
        1.0, 1.0, text,
        transform=ax.transAxes + off,
        ha="right", va="bottom",
        fontsize=fontsize, color=color
    )


# =====================================================
# 4) ax6 数值（固定）
# =====================================================
P_LEVELS_AX6 = np.array([70, 50, 30, 20, 10], dtype=float)

TR_FUB  = np.array([-3.85, -2.58, +0.86, +1.58, -0.57], dtype=float)
ER_FUB  = np.array([ 1.79,  0.66,  0.51,  0.47,  0.88], dtype=float)

TR_ERA5 = np.array([-1.53, -2.49, +1.22, +1.05, -1.54], dtype=float)
ER_ERA5 = np.array([ 1.09,  0.59,  0.54,  0.48,  0.56], dtype=float)


# =====================================================
# 5) 读取数据
# =====================================================
print("Loading SST trend & time series...")
ds_tr = xr.open_dataset(TREND_NC)
OBS_TREND_MME = ds_tr["trend"]
ds_tr.close()

with open(PKL, "rb") as f:
    data = pickle.load(f)

obs_ens_band  = data["obs_ens_band"]
cmip_his_band = data["cmip_his_band"]
cmip_245_band = data["cmip_245_band"]
cmip_585_band = data["cmip_585_band"]
BASELINE      = data["BASELINE"]
SCEN_START    = int(data["SCEN_START"])
HIS_END       = SCEN_START - 1

print("Loading QBO amplitude profile cache (for ax7)...")
ds_qbo = xr.open_dataset(CACHE_NC)
try:
    ds_qbo = ds_qbo.load()
finally:
    ds_qbo.close()


# =====================================================
# 6) 建图
# =====================================================
print("Building figure...")
fig = plt.figure(figsize=(8, 4), dpi=500)

gs = GridSpec(
    3, 3, figure=fig,
    width_ratios=[1, 1, 0.5],
    height_ratios=HEIGHT_RATIOS,
    hspace=HSPACE_MAIN, wspace=0.20,
    left=0.05, right=0.98, top=0.96, bottom=0.04
)

ax1 = fig.add_subplot(gs[0, 0:2], projection=ccrs.PlateCarree(central_longitude=180))
ax2 = fig.add_subplot(gs[1, 0])  # (b)
ax3 = fig.add_subplot(gs[1, 1])  # (c)
ax4 = fig.add_subplot(gs[2, 0])  # (d)
ax5 = fig.add_subplot(gs[2, 1])  # (e)

gs_r = gs[:, 2].subgridspec(2, 1, hspace=FG_HSPACE, height_ratios=[1, 1])
ax6 = fig.add_subplot(gs_r[0, 0])
ax7 = fig.add_subplot(gs_r[1, 0])

ax1.set_aspect("auto")

# 先缩放
shrink_axes_vertically([ax2, ax3, ax4, ax5], shrink_abs=BCDE_SHRINK_HEIGHT)

# 你的原始移动（第三行保持原样）
shift_axes_y([ax2, ax3], dy=-SHIFT_BC_DOWN)
shift_axes_y([ax4, ax5], dy=+SHIFT_DE_UP)

# ✅ 只额外下移第二行(b/c)，第三行不动
shift_axes_y([ax2, ax3], dy=-BC_ONLY_EXTRA_DOWN)

force_align_right_column(ax1, ax4, ax5, ax6, ax7, gap_frac=GAP_FRAC_RIGHT)


# =====================================================
# 7) ax1：SST trend
# =====================================================
print("Plotting ax1...")
pc = ccrs.PlateCarree()
ax1.set_extent([-180, 240, -30, 30], crs=pc)
ax1.set_xticks([0, 60, 120, 180, 240, 300, 359.9999999999], crs=pc)
ax1.set_yticks(np.arange(-30, 30 + 15, 15), crs=pc)
ax1.xaxis.set_major_formatter(LongitudeFormatter())
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax1.add_feature(cfeature.LAND, facecolor="white", zorder=2)
ax1.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=3)

basin_lines = [(-70, "TAO"), (30, "TIO"), (110, "TPO")]
for L, lab in basin_lines:
    L360 = lon_to_0_360(L)
    ax1.plot([L360, L360], [-30, 30], transform=pc,
             color=A_SPLIT_COL, lw=REF_LW_A_BOLD, ls="--", zorder=4)
    ax1.text(
        L360 + 3.5 + A_BASIN_LABEL_DX_DEG,
        28.3 - A_BASIN_LABEL_DY_DEG,
        lab,
        transform=pc, ha="left", va="top",
        fontsize=LABEL_FS_A, color=LABEL_COL_A, fontweight=LABEL_WEIGHT_A,
        bbox=LABEL_BBOX_A,
        zorder=8
    )

try:
    import cmaps
    DIV_CMAP = cmaps.cmp_b2r
except Exception:
    DIV_CMAP = plt.get_cmap("RdBu_r")

levels = np.arange(-0.20, 0.20 + 0.01, 0.02)
data_cyc, lon_cyc = add_cyclic_point(OBS_TREND_MME.values, coord=OBS_TREND_MME["lon"].values)

cf = ax1.contourf(
    lon_cyc, OBS_TREND_MME["lat"].values, np.clip(data_cyc, -0.20, 0.20),
    levels=levels, cmap=DIV_CMAP, transform=pc, extend="neither"
)

ax1.set_title("(a) Tropical SST trend", loc="left", x=TITLE_X, fontsize=TITLE_FS, pad=TITLE_PAD)
ax1.set_xlabel("Longitude", fontsize=XYLABEL_FS, labelpad=X_LABELPAD)
ax1.set_ylabel("Latitude", fontsize=XYLABEL_FS)

add_title_right_text_aligned_to_title(
    ax1, "1950–2020",
    fontsize=TICK_FS, color=COL_OBS,
    pad=TITLE_PAD, dy_pt=A_RIGHT_TEXT_DY_PT
)

pos1 = ax1.get_position()
cb_w, cb_h = 0.38, 0.012
cb_x = (pos1.x0 + pos1.x1) / 2.0 - cb_w / 2.0
cb_y = pos1.y0 + CB_Y_OFFSET
cax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])

cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
cb.set_ticks(levels[::2])
cb.ax.tick_params(labelsize=TICK_FS, width=0.3, length=2.5)
cb.ax.text(
    1.02, 0.5,
    "SST Trend (K decade$^{-1}$)",
    transform=cb.ax.transAxes,
    ha="left", va="center",
    fontsize=XYLABEL_FS
)


# =====================================================
# 8) ax2-ax5：b-e
# =====================================================
print("Plotting ax2-ax5...")
axes = {"TO": ax2, "TIO": ax3, "TAO": ax4, "TPO": ax5}
title_txt = {
    "TO":  "(b) TO SST Anomaly",
    "TIO": "(c) TIO SST Anomaly",
    "TAO": "(d) TAO SST Anomaly",
    "TPO": "(e) TPO SST Anomaly",
}
XTICKS_BE = [1950, 1980, 2010, 2040, 2070, 2100]
XMIN, XMAX = 1950, 2100

for r in ["TO", "TIO", "TAO", "TPO"]:
    ax = axes[r]
    close_box_axes(ax)

    ax.axhline(0, color=REF_COL, lw=REF_LW, ls="--")
    ax.axvline(SCEN_START, color=REF_COL, lw=REF_LW, ls="--")

    ax.text(0.98, 0.14, f"relative to {BASELINE[0]}–{BASELINE[1]}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, fontstyle="italic", color=REF_COL)

    if r in cmip_his_band:
        med, lo, hi = cmip_his_band[r]
        draw_band(ax, med.sel(year=slice(None, HIS_END)),
                  lo.sel(year=slice(None, HIS_END)),
                  hi.sel(year=slice(None, HIS_END)),
                  COL_HIS, ALPHA_HIS, LW_ALL, "Historical")

    if r in cmip_245_band:
        med, lo, hi = cmip_245_band[r]
        draw_band(ax, med.sel(year=slice(SCEN_START, None)),
                  lo.sel(year=slice(SCEN_START, None)),
                  hi.sel(year=slice(SCEN_START, None)),
                  COL_245, ALPHA_245, LW_ALL, "SSP2-4.5")

    if r in cmip_585_band:
        med, lo, hi = cmip_585_band[r]
        draw_band(ax, med.sel(year=slice(SCEN_START, None)),
                  lo.sel(year=slice(SCEN_START, None)),
                  hi.sel(year=slice(SCEN_START, None)),
                  COL_585, ALPHA_585, LW_ALL, "SSP5-8.5")

    med, lo, hi = obs_ens_band[r]
    draw_band(ax, med, lo, hi, COL_OBS, ALPHA_OBS, LW_ALL, "Observation-based", hatch="////")

    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks(XTICKS_BE)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.margins(x=0)

    ax.set_ylim(-1, 7)
    ax.set_yticks([0, 2, 4, 6])

    if r in ["TIO", "TPO"]:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("SST Anomaly (K)", fontsize=XYLABEL_FS)

    ax.set_title(title_txt[r], loc="left", x=TITLE_X, fontsize=TITLE_FS, pad=TITLE_PAD)
    add_ordered_legend_be(ax)

    if r in ["TAO", "TPO"]:
        ax.set_xlabel("Year", fontsize=XYLABEL_FS, labelpad=X_LABELPAD)


# =====================================================
# 9) ax6：QBO trend
# =====================================================
print("Plotting ax6...")
close_box_axes(ax6)
ax6.axvline(0, color=REF_COL, lw=REF_LW, ls="--", zorder=0)

ax6.errorbar(TR_FUB,  P_LEVELS_AX6, xerr=ER_FUB, fmt="none",
             ecolor=COL_FUB_F, elinewidth=LW_ALL, alpha=0.20, zorder=1)
ax6.errorbar(TR_ERA5, P_LEVELS_AX6, xerr=ER_ERA5, fmt="none",
             ecolor=COL_ERA5_F, elinewidth=LW_ALL, alpha=0.25, zorder=1)

ax6.plot(TR_FUB,  P_LEVELS_AX6, color=COL_FUB_F,  lw=LW_ALL, zorder=3, label="FUB")
ax6.plot(TR_ERA5, P_LEVELS_AX6, color=COL_ERA5_F, lw=LW_ALL, zorder=3, label="ERA5")

mask = np.ones_like(P_LEVELS_AX6, dtype=bool); mask[0] = False
ax6.scatter(TR_FUB[mask],  P_LEVELS_AX6[mask],  s=52, color=COL_FUB_F,  zorder=4,
            edgecolors="white", linewidths=0.6)
ax6.scatter(TR_ERA5[mask], P_LEVELS_AX6[mask], s=52, color=COL_ERA5_F, zorder=4,
            edgecolors="white", linewidths=0.6)

set_pressure_axis_integer_ticks(ax6, ticks=(70, 50, 30, 20, 10), ylim_top=75, ylim_bottom=9)
ax6.set_xlim(-6.2, 2.2)
ax6.set_xticks([-6, -4, -2, 0, 2])

ax6.set_xlabel("QBO Amplitude Trend (% decade$^{-1}$)", fontsize=XYLABEL_FS, labelpad=X_LABELPAD)
ax6.set_ylabel("Pressure (hPa)", fontsize=XYLABEL_FS)

ax6.set_title("(f) QBO amplitude trend", loc="left", x=TITLE_X, fontsize=TITLE_FS, pad=TITLE_PAD)
ax6.text(0.99, 0.04, "1956–2020",
         transform=ax6.transAxes, ha="right", va="bottom",
         fontsize=TICK_FS, color=COL_OBS)

h, l = ax6.get_legend_handles_labels()
order = ["FUB", "ERA5"]
mp = {lab: hh for hh, lab in zip(h, l)}
ax6.legend([mp[k] for k in order if k in mp],
           [k for k in order if k in mp],
           loc="upper left", frameon=False, fontsize=LEGEND_FS,
           handlelength=LEG_HANDLELENGTH, borderaxespad=LEG_BORDERPAD,
           labelspacing=LEG_LABELSPACING)
ax6.grid(False)


# =====================================================
# 10) ax7：QBO amplitude profile
# =====================================================
print("Plotting ax7...")
close_box_axes(ax7)

p_cmip = ds_qbo["press_cmip_hPa"].values
h_mean = ds_qbo["mmm_hist_1950_2014_mean"].values
h_2se  = ds_qbo["mmm_hist_1950_2014_2se"].values
p245_mean = ds_qbo["mmm_ssp245_2015_2100_mean"].values
p245_2se  = ds_qbo["mmm_ssp245_2015_2100_2se"].values
p585_mean = ds_qbo["mmm_ssp585_2015_2100_mean"].values
p585_2se  = ds_qbo["mmm_ssp585_2015_2100_2se"].values
era5_main = ds_qbo["era5_main_1950_2020_on_cmip"].values

line_his, = ax7.plot(h_mean, p_cmip, lw=LW_ALL, color=COL_HIS, label="Historical")
lo, hi = np.maximum(h_mean - h_2se, 0.0), h_mean + h_2se
ax7.fill_betweenx(p_cmip, lo, hi, alpha=0.18, color=COL_HIS, linewidth=0)

line_245 = None
if np.isfinite(p245_mean).any():
    line_245, = ax7.plot(p245_mean, p_cmip, lw=LW_ALL, color=COL_245, label="SSP2-4.5")
    lo, hi = np.maximum(p245_mean - p245_2se, 0.0), p245_mean + p245_2se
    ax7.fill_betweenx(p_cmip, lo, hi, alpha=0.14, color=COL_245, linewidth=0)

line_585 = None
if np.isfinite(p585_mean).any():
    line_585, = ax7.plot(p585_mean, p_cmip, lw=LW_ALL, color=COL_585, label="SSP5-8.5")
    lo, hi = np.maximum(p585_mean - p585_2se, 0.0), p585_mean + p585_2se
    ax7.fill_betweenx(p_cmip, lo, hi, alpha=0.14, color=COL_585, linewidth=0)

line_era5, = ax7.plot(era5_main, p_cmip, lw=LW_ALL, color=COL_ERA5, label="ERA5")

set_pressure_axis_integer_ticks(ax7, ticks=(70, 50, 30, 20, 10), ylim_top=75, ylim_bottom=9)
ax7.set_xlim(0, 16)
ax7.set_xticks([0, 4, 8, 12, 16])

ax7.set_xlabel("QBO Fourier amplitude (m s$^{-1}$)", fontsize=XYLABEL_FS, labelpad=X_LABELPAD)
ax7.set_ylabel("Pressure (hPa)", fontsize=XYLABEL_FS)

ax7.set_title("(g) QBO amplitude", loc="left", x=TITLE_X, fontsize=TITLE_FS, pad=TITLE_PAD)
ax7.text(0.99, 0.03, "(20–40 months)",
         transform=ax7.transAxes, ha="right", va="bottom",
         fontsize=TICK_FS, color=COL_OBS)

handles = [line_era5, line_his]
labels  = ["ERA5", "Historical"]
if line_245 is not None:
    handles.append(line_245); labels.append("SSP2-4.5")
if line_585 is not None:
    handles.append(line_585); labels.append("SSP5-8.5")
ax7.legend(handles, labels, loc="upper left",
           frameon=False, fontsize=LEGEND_FS,
           handlelength=LEG_HANDLELENGTH, borderaxespad=LEG_BORDERPAD,
           labelspacing=LEG_LABELSPACING)


# =====================================================
# 11) f/g：左右 y 都显示，ylabel 放右侧
# =====================================================
if MOVE_FG_Y_TO_RIGHT:
    keep_left_and_add_right_y(ax6)
    keep_left_and_add_right_y(ax7)


# ==================================================
