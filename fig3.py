import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

purple_seq = ["#e6effb", "#b9ceff", "#83acf9", "#4f8ffc", "#0e72fc"]
red_seq = ["#ffffb2ff", "#fecc5cff", "#fd8d3cff", "#f03b20ff", "#bd0026ff"]
diverging_colors = list(reversed(purple_seq)) + red_seq
div_cmap = LinearSegmentedColormap.from_list(
    "purple_to_red_diverging", diverging_colors, N=256
)

def truncate_cmap(cmap, vmin=0.0, vmax=1.0, n=256):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(vmin, vmax, n))
    return ListedColormap(colors)

abs_cmap = truncate_cmap(div_cmap, 0.5, 1.0, n=256)

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 7
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5

file_paths = {
    'CTRL': "TS.CTRL2000.nc",
    'TO': "TS.TO2000.nc",
    'TIO': "TS.TIO2000.nc",
    'TAO': "TS.TAO2000.nc",
    'TPO': "TS.TPO2000.nc"
}

ts_means = {}
for name, path in file_paths.items():
    ds = xr.open_dataset(path)
    ts_means[name] = ds['TS'].mean(dim='time')
    ds.close()

lat = ts_means['CTRL']['lat'].values
lon = ts_means['CTRL']['lon'].values
CTRL = ts_means['CTRL']

proj = ccrs.PlateCarree(central_longitude=180)
fig, axes = plt.subplots(
    nrows=5, ncols=1,
    figsize=(6.8 / 2.54, 10.6 / 2.54),
    subplot_kw={'projection': proj},
    dpi=500
)

plot_configs = [
    ('CTRL', '(a) CTRL', CTRL, 'abs'),
    ('TO', '(b) TO', ts_means['TO'] - CTRL, 'diff'),
    ('TIO', '(c) TIO', ts_means['TIO'] - CTRL, 'diff'),
    ('TAO', '(d) TAO', ts_means['TAO'] - CTRL, 'diff'),
    ('TPO', '(e) TPO', ts_means['TPO'] - CTRL, 'diff'),
]

ABS_LEVELS = np.arange(290, 311, 2)
DIFF_LEVELS = np.arange(-2.0, 2.01, 0.2)

im_abs = None
im_diff = None

for ax, (_, title, data, kind) in zip(axes, plot_configs):
    ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-30, 31, 15), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))

    d_cyc, lon_cyc = add_cyclic_point(data.values, coord=lon)

    if kind == 'abs':
        im_abs = ax.contourf(
            lon_cyc, lat, d_cyc,
            levels=ABS_LEVELS,
            cmap=abs_cmap,
            transform=ccrs.PlateCarree(),
            extend='both'
        )
    else:
        im_diff = ax.contourf(
            lon_cyc, lat, d_cyc,
            levels=DIFF_LEVELS,
            cmap=div_cmap,
            transform=ccrs.PlateCarree(),
            extend='both'
        )

    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    ax.set_title(title, loc='left')

    ax.set_ylabel("Latitude")
    if title.startswith("(e)"):
        ax.set_xlabel("Longitude")

if im_abs is not None:
    cbar_abs = fig.colorbar(im_abs, ax=axes[0], orientation='vertical')
    cbar_abs.set_label('SST (K)')
    cbar_abs.set_ticks([290, 295, 300, 305, 310])

if im_diff is not None:
    cbar_diff = fig.colorbar(im_diff, ax=axes[1:], orientation='vertical')
    cbar_diff.set_label('ΔSST (K)')

fig.savefig("Figure3.pdf", format="pdf", dpi=500, bbox_inches="tight")

plt.show()
