# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

era_years = dates.year.values[:504]

year_ticks = np.arange(1980, 2022, 5)
year_positions = []
for year in year_ticks:
    pos = np.where(era_years == year)[0]
    if len(pos) > 0:
        year_positions.append(int(pos[0]))

p_ticks = [100, 70, 50, 30, 20, 10, 7, 5]
lat_ticks = [-60, -30, 0, 30, 60]
lat_labels = ['60°S', '30°S', '0°', '30°N', '60°N']

fig = plt.figure(figsize=(6, 6), dpi=500)
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], hspace=0.6)
plt.subplots_adjust(wspace=0.20)

ax0 = fig.add_subplot(gs[0, :])
quad = ax0.contourf(
    time, press_era, um0.T,
    levels=np.arange(-36, 36.1, 6),
    cmap=div_cmap, extend='both'
)
ax0.contour(
    time, press_ctrl, um1.T,
    levels=np.arange(-35, 40, 5),
    colors='k', linewidths=0.5
)
ax0.contour(
    time, press_ctrl, um1.T,
    levels=[0], colors='k', linewidths=1.5
)
ax0.set_yscale('log')
ax0.set_ylim(100, 5)
ax0.set_ylabel('Pressure (hPa)')
ax0.set_yticks(p_ticks)
ax0.set_yticklabels([str(v) for v in p_ticks])
ax0.tick_params(axis='y', which='minor', length=0)
ax0.set_xticks(year_positions)
ax0.set_xticklabels([str(y) for y in year_ticks])
ax0.set_xlabel('Year')
ax0.set_title('(a) Equatorial zonal-mean zonal wind', loc='left', pad=6)

ax1 = fig.add_subplot(gs[1, 0])
cf1 = ax1.contourf(
    era['freqs'], press_era, era['spec'],
    levels=levels_spec, cmap=red_cmap, extend='both'
)
ax1.axvline(era['pmin'], ls='--', color='black')
ax1.axvline(era['pmax'], ls='--', color='black')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(12, 0.25)
ax1.set_ylim(100, 5)
ax1.set_ylabel('Pressure (hPa)')
ax1.set_xlabel('Period (years)')
ax1.set_yticks(p_ticks)
ax1.set_yticklabels([str(v) for v in p_ticks])
ax1.tick_params(axis='y', which='minor', length=0)
ax1.set_xticks([0.5, 1, 2, 3, 5, 10])
ax1.set_xticklabels([0.5, 1, 2, 3, 5, 10])
ax1.set_title('(b) ERA5 QBO spectra', loc='left', pad=6)

ax2 = fig.add_subplot(gs[1, 1])
cf2 = ax2.contourf(
    lat_era, press_era, era['amp_lp'].T,
    levels=levels_amp, cmap=red_cmap, extend='both'
)
ax2.contour(
    lat_era, press_era, era['amp_lp'].T,
    levels=levels_amp, colors='k', linewidths=0.5
)
ax2.set_yscale('log')
ax2.set_ylim(100, 5)
ax2.set_xlim(-60, 60)
ax2.set_yticks(p_ticks)
ax2.set_yticklabels([str(v) for v in p_ticks])
ax2.tick_params(axis='y', which='minor', length=0)
ax2.set_xticks(lat_ticks)
ax2.set_xticklabels(lat_labels)
ax2.set_xlabel('Latitude')
ax2.set_title('(c) ERA5 QBO Fourier amplitude', loc='left', pad=6)

ax3 = fig.add_subplot(gs[2, 0])
cf3 = ax3.contourf(
    ctrl['freqs'], press_ctrl, ctrl['spec'],
    levels=levels_spec, cmap=purple_cmap, extend='both'
)
ax3.axvline(ctrl['pmin'], ls='--', color='black')
ax3.axvline(ctrl['pmax'], ls='--', color='black')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim(12, 0.25)
ax3.set_ylim(100, 5)
ax3.set_xlabel('Period (years)')
ax3.set_ylabel('Pressure (hPa)')
ax3.set_yticks(p_ticks)
ax3.set_yticklabels([str(v) for v in p_ticks])
ax3.tick_params(axis='y', which='minor', length=0)
ax3.set_xticks([0.5, 1, 2, 3, 5, 10])
ax3.set_xticklabels([0.5, 1, 2, 3, 5, 10])
ax3.set_title('(d) CTRL QBO spectra', loc='left', pad=6)

ax4 = fig.add_subplot(gs[2, 1])
cf4 = ax4.contourf(
    lat_ctrl, press_ctrl, ctrl['amp_lp'].T,
    levels=levels_amp, cmap=purple_cmap, extend='both'
)
ax4.contour(
    lat_ctrl, press_ctrl, ctrl['amp_lp'].T,
    levels=levels_amp, colors='k', linewidths=0.5
)
ax4.set_yscale('log')
ax4.set_ylim(100, 5)
ax4.set_xlim(-60, 60)
ax4.set_yticks(p_ticks)
ax4.set_yticklabels([str(v) for v in p_ticks])
ax4.tick_params(axis='y', which='minor', length=0)
ax4.set_xticks(lat_ticks)
ax4.set_xticklabels(lat_labels)
ax4.set_xlabel('Latitude')
ax4.set_title('(e) CTRL QBO Fourier amplitude', loc='left', pad=6)

cb_width = 0.010
pad = 0.010

pos_a = ax0.get_position()
cax_a = fig.add_axes([pos_a.x1 + pad, pos_a.y0, cb_width, pos_a.height])
cbar_a = fig.colorbar(quad, cax=cax_a, extendrect=True)
cbar_a.set_label('U (m s$^{-1}$)')

pos_b = ax1.get_position()
cax_b = fig.add_axes([pos_b.x1 + pad, pos_b.y0, cb_width, pos_b.height])
cbar_b = fig.colorbar(cf1, cax=cax_b, extendrect=True)
cbar_b.set_label('Spectral magnitude')

pos_c = ax2.get_position()
cax_c = fig.add_axes([pos_c.x1 + pad, pos_c.y0, cb_width, pos_c.height])
cbar_c = fig.colorbar(cf2, cax=cax_c, extendrect=True)
cbar_c.set_label('Amplitude (m s$^{-1}$)')

pos_d = ax3.get_position()
cax_d = fig.add_axes([pos_d.x1 + pad, pos_d.y0, cb_width, pos_d.height])
cbar_d = fig.colorbar(cf3, cax=cax_d, extendrect=True)
cbar_d.set_label('Spectral magnitude')

pos_e = ax4.get_position()
cax_e = fig.add_axes([pos_e.x1 + pad, pos_e.y0, cb_width, pos_e.height])
cbar_e = fig.colorbar(cf4, cax=cax_e, extendrect=True)
cbar_e.set_label('Amplitude (m s$^{-1}$)')

plt.show()
