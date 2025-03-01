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

# =================================================================================== #
# ----------------------- ------- UTILITY FUNCTION ---------------------------------- #
# =================================================================================== #

def correct_circle_diff(
                         angle1     : float,
                         angle2     : float
                       ) -> float:
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

    # Return correction
    print("YI", min(ang_diff, ang_diff + 2*np.pi, ang_diff - 2*np.pi, key=abs))
    return min(ang_diff, ang_diff + 2*np.pi, ang_diff - 2*np.pi, key=abs)


# =================================================================================== #
# -------------------------------- MAIN FUNCTION ------------------------------------ #
# =================================================================================== #

def calculate_ROE(
                    oe          : np.ndarray,
                    oed         : np.ndarray,
                    set_normal  : bool = True
                 )  -> np.ndarray:
    """
    Calculates the circular, quasi-nonsingular relative orbital elements (ROE)
    using the circular, Keplerian orbital elements (uses MA instead of TA).

    This computes the NORMALIZED version of the ROEs.

    Here, it assumed that the order of orbital elements are as follows:

    ->  oe = [a, e, i, AOP, RAAN, MA]
    ->  Units: km, -, deg, deg, deg, deg

    Args:
        oe (np.ndarray):        Orbital elements of primary spacecraft (Client)
        oed (np.ndarray):       Orbital elements of secondary spacecraft (Servicer)
        set_normal (bool):      Whether to normalize the elements. Default is True.

    Returns:
        np.ndarray:             Normalized/non-normalized relative orbital elements
    """
 
    # Extract elements from primary s/c
    a = oe[0]
    ecc = oe[1]
    inc = np.radians(oe[2])
    aop = np.radians(oe[3])
    raan = np.radians(oe[4])
    ma = np.radians(oe[5])

    # Extract elements from secondary s/c
    ad = oed[0]
    eccd = oed[1]
    incd = np.radians(oed[2])
    aopd = np.radians(oed[3])
    raand = np.radians(oed[4])
    mad = np.radians(oed[5])

    # Compute angular differences
    aopd_minus_aop = correct_circle_diff(aop, aopd)
    mad_minus_ma = correct_circle_diff(ma, mad)
    raand_minus_raan = correct_circle_diff(raan, raand)

    # Compute relative orbital elements
    da = (ad - a)/a
    dl = mad_minus_ma + aopd_minus_aop + raand_minus_raan*np.cos(inc)
    dex = eccd*np.cos(aopd) - ecc*np.cos(aop)
    dey = eccd*np.sin(aopd) - ecc*np.sin(aop)
    dix = incd - inc
    diy = (raand_minus_raan)*np.sin(inc)

    # Return output
    return np.array([da, dl, dex, dey, dix, diy], dtype=float) if set_normal else a*np.array([da, dl, dex, dey, dix, diy], dtype=float)