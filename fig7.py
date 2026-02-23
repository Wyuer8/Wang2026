# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


pressures = np.array([70.0, 50.0, 30.0, 20.0, 10.0])

data_profile = {
    "CTRL": np.array([0.154559, 0.146825, 0.233569, 0.354083, 0.502748]),
    "TO"  : np.array([0.163306, 0.144798, 0.223255, 0.351380, 0.480827]),
    "TIO" : np.array([0.142684, 0.131621, 0.231915, 0.338000, 0.464460]),
    "TAO" : np.array([0.156084, 0.145043, 0.232002, 0.347844, 0.467538]),
    "TPO" : np.array([0.142382, 0.129415, 0.220642, 0.351201, 0.552065]),
}

phases = np.arange(1, 10)

data_phase10 = {
    "CTRL": np.array([0.498, 0.674, 0.808, 0.687, 0.587, 0.358, 0.112, 0.179, 0.498]),
    "TO"  : np.array([0.523, 0.733, 0.733, 0.655, 0.619, 0.323, 0.134, 0.114, 0.523]),
    "TIO" : np.array([0.421, 0.665, 0.736, 0.592, 0.619, 0.326, 0.119, 0.179, 0.421]),
    "TAO" : np.array([0.529, 0.746, 0.697, 0.602, 0.585, 0.254, 0.019, 0.221, 0.529]),
    "TPO" : np.array([0.481, 0.594, 0.724, 0.725, 0.719, 0.581, 0.401, 0.298, 0.481]),
}

colors = {
    "CTRL": "#34314c",
    "TO"  : "#47b8e0",
    "TIO" : "#f26d5b",
    "TAO" : "#FBA518",
    "TPO" : "#6CA651",
}

order = ["CTRL", "TO", "TIO", "TAO", "TPO"]

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=500)

ax0 = axes[0]
for name in order:
    ax0.plot(data_profile[name], pressures, color=colors[name], label=name)

ax0.invert_yaxis()
ax0.set_yscale("symlog")
ax0.set_ylim(70, 10)
ax0.set_yticks([70, 50, 30, 20, 10])
ax0.set_yticklabels([70, 50, 30, 20, 10])
ax0.set_xlabel(r"$w^*$ (mm s$^{-1}$)")
ax0.set_ylabel("Pressure (hPa)")

ax1 = axes[1]
for name in order:
    ax1.plot(phases, data_phase10[name], color=colors[name], label=name)

ax1.set_xlim(1, 9)
ax1.set_xticks(phases)
ax1.set_ylim(0, 1.0)
ax1.set_yticks(np.arange(0, 1.01, 0.2))
ax1.set_xlabel("Phase")
ax1.set_ylabel(r"$w^*$ (mm s$^{-1}$)")

ax1.legend(frameon=False)

out_dir = os.getenv("FIG_OUT_DIR", ".")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "Figure7.pdf")
fig.savefig(out_path, format="pdf", bbox_inches="tight")

print(f"Figure saved to: {out_path}")

plt.show()
