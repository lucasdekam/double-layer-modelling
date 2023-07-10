"""
Functions for converting PotentialSweepSolutions into reaction currents.
"""
import numpy as np
import edl
import constants as C

def frumkin_corrected_current(model: edl.DoubleLayerModel,
                            potential_range_she: np.ndarray,
                            pzc_she: float,
                            p_h: float,
                            include_cat: bool=True) -> np.ndarray:
    """
    Compute the Frumkin-corrected current:
    j ~ exp(-alpha * f * |E(D_ADSORBATE_LAYER)| * D_ADSORBATE_LAYER)
    (f=F/RT=beta*e0; assume alpha=0.5)

    If include_cat is True, the current is also linearly proportional to the
    cation concentration at x=D_ADSORBATE_LAYER:
    j ~ exp(-alpha * f * |E(D_ADSORBATE_LAYER)| * D_ADSORBATE_LAYER) * [Cat]

    We take the current to be negative (cathodic; reduction reaction)

    model: double layer model
    potential_range_she: potential range in V vs. SHE
    pzc_she: potential of zero charge of the metal in V vs. SHE
    """
    sol = model.potential_sweep(potential_range_she - pzc_she, p_h=p_h)

    efield = np.zeros(potential_range_she.shape)
    c_cat = np.ones(potential_range_she.shape)
    c_h2o = np.ones(potential_range_she.shape)
    for i, _ in enumerate(potential_range_she):
        efield[i] = sol.profiles[i].efield[0]
        c_h2o[i] = sol.profiles[i].c_dict[r'Solvent'][0]
        if include_cat:
            c_cat[i] = sol.profiles[i].c_dict[r'Cations'][0]

    current = - np.exp(-0.5*C.E_0*C.BETA * C.D_ADSORBATE_LAYER * efield) * c_h2o
    return current

def edl_transport_limited_current(model: edl.DoubleLayerModel,
                                   potential_range_she: np.ndarray,
                                   pzc_she: float,
                                   p_h: float) -> np.ndarray:
    """
    Compute the current that is rate-limited by transport away from the
    electrode surface:
    j ~ exp(-f * phi2)
    (f=F/RT=beta*e0)

    We take the current to be negative (cathodic; reduction reaction)

    model: double layer model
    potential_range_she: potential range in V vs. SHE
    pzc_she: potential of zero charge of the metal in V vs. SHE
    """
    sol = model.potential_sweep(potential_range_she - pzc_she, p_h=p_h)

    phi_2 = np.zeros(potential_range_she.shape)
    c_h2o = np.ones(potential_range_she.shape)
    efield = np.zeros(potential_range_she.shape)

    for i, _ in enumerate(potential_range_she):
        efield[i] = sol.profiles[i].efield[0]
        c_h2o[i] = sol.profiles[i].c_dict[r'Solvent'][0]
        phi_2[i] = sol.profiles[i].phi[0]

    current = -np.exp(-C.E_0*C.BETA * phi_2)
    # current = - np.exp(-0.5*C.E_0*C.BETA * C.D_ADSORBATE_LAYER * efield) * c_h2o

    return current
