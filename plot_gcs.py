"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

from edl import models
from edl import constants as C
import plotting as P

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

PHI0_V = -1


def stern(x, slope):  # pylint: disable=invalid-name
    """
    Potential profile in Stern layer
    """
    return PHI0_V - slope * x


potentials = np.linspace(-1, 1, 100)
concentration_range = P.CONC_LIST

capa_phi_sweep = np.zeros((len(concentration_range), len(potentials)))
sigma_phi_sweep = np.zeros((len(concentration_range), len(potentials)))
x_axes = []
phi_spatial = []
efield_spatial = []
cation_spatial = []
anion_spatial = []

for i, conc in enumerate(concentration_range):
    gc = models.GouyChapmanStern(conc)
    gc_phi_sweep = gc.potential_sweep(potentials, tol=1e-3)
    capa_phi_sweep[i, :] = gc_phi_sweep["capacity"]
    sigma_phi_sweep[i, :] = gc_phi_sweep["charge"]

    gc_spatial = gc.spatial_profiles(phi0=PHI0_V, tol=1e-3)
    x_axes.append(gc_spatial["x"] + C.D_ADSORBATE_LAYER * 1e9)
    phi_spatial.append(gc_spatial["phi"])
    efield_spatial.append(gc_spatial["efield"])
    cation_spatial.append(gc_spatial["cations"])
    anion_spatial.append(gc_spatial["anions"])


fig, ax = plt.subplots(figsize=(5, 4), nrows=2, ncols=2)
colors1 = P.get_color_gradient(len(concentration_range))
colors2 = P.get_color_gradient(len(concentration_range), color="red")

# c_entries = []
# a_entries = []

for i, conc in enumerate(concentration_range):
    ax[0, 0].plot(
        x_axes[i],
        phi_spatial[i],
        label=f"{conc*1e3:.0f}",
        color=colors1[i],
    )
    x_m = np.linspace(0, C.D_ADSORBATE_LAYER, 10)

    ax[0, 0].plot(
        x_m * 1e9,
        stern(x_m, efield_spatial[i][0]),
        color=colors1[i],
    )

    (c,) = ax[0, 1].plot(
        x_axes[0],
        cation_spatial[i],
        label=f"{conc*1e3:.0f} mM",
        color=colors1[i],
    )
    # (a,) = ax[0, 1].plot(
    #     x_axes[0],
    #     anion_spatial[i],
    #     label=f"{conc*1e3:.0f} mM",
    #     color=colors2[i],
    # )
    # c_entries.append(c)
    # a_entries.append(a)
    ax[1, 0].plot(
        potentials,
        sigma_phi_sweep[i, :] * 100,
        label=f"{conc*1e3:.0f} mM",
        color=colors1[i],
    )
    ax[1, 1].plot(
        potentials,
        capa_phi_sweep[i, :] * 100,
        label=f"{conc*1e3:.0f} mM",
        color=colors1[i],
    )

ax[0, 0].set_ylabel(r"$\phi$ / V")
ax[0, 1].set_ylabel(r"$c_+$ / M")
ax[1, 0].set_ylabel(r"$\sigma$ / $\mu$C cm$^{-2}$")
ax[1, 1].set_ylabel(r"$C$ / $\mu$F cm$^{-2}$")

ax[0, 0].set_ylim([PHI0_V, 0.05])
# ax[0, 1].set_ylim([1e-3, 1e4])
ax[1, 0].set_ylim([-200, 200])
ax[1, 1].set_ylim([0, 300])

ax[0, 0].set_xlabel(r"$x$ / nm")
ax[0, 1].set_xlabel(r"$x$ / nm")
ax[1, 0].set_xlabel(r"$\phi_0$ / V")
ax[1, 1].set_xlabel(r"$\phi_0$ / V")

ax[0, 0].set_xlim([0, 2])
ax[0, 1].set_xlim([0, 2])
ax[1, 0].set_xlim([potentials[0], potentials[-1]])
ax[1, 1].set_xlim([potentials[0], potentials[-1]])

# ax[0, 0].set_xticks([0, 1, 2, 3, 4, 5])
# ax[0, 1].set_xticks([0, 1, 2, 3, 4, 5])

ax[0, 0].legend(frameon=False, title=r"$c_0$ / mM")

# ax[0, 1].set_yscale("log")
# leg2 = ax[0, 1].legend(
#     [(c, a) for (c, a) in zip(c_entries, a_entries)],
#     [f"{conc*1e3:.0f} mM" for conc in concentration_range],
#     handler_map={tuple: lh.HandlerTuple(ndivide=None)},
#     handlelength=5,
#     title=r"  $+$      $-$   ",
#     alignment="left",
#     frameon=False,
# )
# ax[0, 1].add_artist(leg2)

labels = ["(a)", "(b)", "(c)", "(d)"]
for label, axis in zip(labels, ax.reshape(-1)):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-25 / 72, 10 / 72, fig.dpi_scale_trans)
    axis.text(
        0.0,
        1.0,
        label,
        transform=axis.transAxes + trans,
        fontsize="medium",
        va="bottom",
    )

plt.tight_layout()
plt.savefig("figures/intro-gouy-chapman-stern.pdf")
plt.show()
