''''

Classical/Keplerian Orbital Elements to Relative Orbital Elements

AUTHOR: Bruce L. Barbour, 2023
        Virginia Tech

This Python script supplies the function to convert the Keplerian orbital elements
to the quasi-nonsingular Relative Orbital Elements (ROE).
'''

# =================================================================================== #
# ------------------------------- IMPORT PACKAGES ----------------------------------- #
# =================================================================================== #

from __future__ import annotations
import numpy as np
from numpy import pi, cos, sin, array
from numba import jit, f8

# =================================================================================== #
# ----------------------- ------- UTILITY FUNCTION ---------------------------------- #
# =================================================================================== #
@jit(f8(f8, f8), nopython=True, fastmath=True)
def correct_circle_diff(angle1, angle2):
    """
    Computes the corrected difference by taking into account the circular nature of angles.

    Args:
        angle1 (float):     Reference angle, in radians
        angle2 (float):     Second angle, in radians

    Returns:
        float:              Corrected difference
    """

    # Compute angular difference
    ang_diff = angle2 - angle1

    # Compute corrected differences
    diff1 = ang_diff
    diff2 = ang_diff + 2 * pi
    diff3 = ang_diff - 2 * pi

    # Precompute absolute differences
    abs_diff1 = abs(diff1)
    abs_diff2 = abs(diff2)
    abs_diff3 = abs(diff3)

    # Return the minimum absolute difference
    if abs_diff1 <= abs_diff2 and abs_diff1 <= abs_diff3:
        return diff1
    elif abs_diff2 <= abs_diff3:
        return diff2
    else:
        return diff3

# =================================================================================== #
# -------------------------------- MAIN FUNCTION ------------------------------------ #
# =================================================================================== #
@jit(f8[:](f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), nopython=True, fastmath=True)
def calculate_ROE(ac, ec, ic, oc, Oc, Mc, ad, ed, id, od, Od, Md):
    """
    Calculates the circular, quasi-nonsingular relative orbital elements (ROE)
    using the circular, Keplerian orbital elements (uses MA instead of TA).

    This computes the NORMALIZED version of the ROEs.

    Here, it assumed that the order of orbital elements are as follows:

    ->  oe = [a, e, i, AOP, RAAN, MA]
    ->  Units: km, -, deg, deg, deg, deg

    Args:
        Orbital elements of primary spacecraft (Client)
        Orbital elements of secondary spacecraft (Servicer)

    Returns:
        np.ndarray: Relative orbital elements
    """

    # Compute angular differences
    aopd_minus_aop = correct_circle_diff(oc, od)
    mad_minus_ma = correct_circle_diff(Mc, Md)
    raand_minus_raan = correct_circle_diff(Oc, Od)

    # Compute relative orbital elements
    da = (ad - ac)/ac
    dl = mad_minus_ma + aopd_minus_aop + raand_minus_raan*cos(ic)
    dex = ed*cos(od) - ec*cos(oc)
    dey = ed*sin(od) - ec*sin(oc)
    dix = id - ic
    diy = (raand_minus_raan)*sin(ic)

    # Return output
    return array([da, dl, dex, dey, dix, diy])


# import time
# oec = np.array([7200, 0.0001, 0.1, 0.001, 0.001, 0.1])
# oed = np.array([7200, 0.0001, 0.1, 0.001, 0.001, 0.1])
# num_run = 1000000

# t = [0., ] * num_run
# for i in range(num_run):
#     calculate_ROE(oec[0], oec[1], oec[2], oec[3], oec[4], oec[5], oed[0], oed[1], oed[2], oed[3], oed[4], oed[5])
#     t0 = time.perf_counter_ns()
#     calculate_ROE(oec[0], oec[1], oec[2], oec[3], oec[4], oec[5], oed[0], oed[1], oed[2], oed[3], oed[4], oed[5])
#     tf = time.perf_counter_ns()
#     t[i] = tf - t0

# print(np.mean(t) * 1e-3)