# =================================================================================== #
# ------------------------------- IMPORT PACKAGES ----------------------------------- #
# =================================================================================== #
import numpy as np
from numba import jit, f8, b1
from numpy import sin, cos, sqrt, array, arctan2, pi

# =================================================================================== #
# ------------------------------ UTILITY FUNCTION ----------------------------------- #
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

@jit(f8[:](f8, f8, f8, f8), nopython=True, fastmath=True)
def envlpparam2coeffs(A, B, Psi, K):
    """
    Computes the polynomial coefficients of the projected envelope based on its ellipse parameters.
    
    Args:
        A (float): Semi-major axis of the ellipse.
        B (float): Semi-minor axis of the ellipse.
        Psi (float): Rotation angle in radians.
        K (float): Offset or scaling factor.

    Returns:
        np.ndarray: Coefficients of the envelope polynomial equation.
    """
    
    # Trigonometric components
    cos_psi = cos(Psi)
    sin_psi = sin(Psi)
    cos_psi_sq = cos_psi**2
    sin_psi_sq = sin_psi**2
    sin_2psi = 2 * sin_psi * cos_psi  # sin(2*Psi)

    # Inversion components
    inv_a2 = 1 / (A**2)
    inv_b2 = 1 / (B**2)

    # Coefficients
    c5 = cos_psi_sq * inv_a2 + sin_psi_sq * inv_b2
    c4 = sin_2psi * (inv_a2 - inv_b2)
    c3 = sin_psi_sq * inv_a2 + cos_psi_sq * inv_b2
    c2 = K * sin_2psi * (inv_b2 - inv_a2)
    c1 = -2 * K * c3
    c0 = -1 + K**2 * c3

    return array([c5, c4, c3, c2, c1, c0])

# =================================================================================== #
# -------------------------------- MAIN FUNCTION ------------------------------------ #
# =================================================================================== #
@jit(b1(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), nopython=True, fastmath=True)
def analytical_passive_safety(AS, BS, SMA, OE1, OE2, OE3, OE4, OE5, OE6, OED1, OED2, OED3, OED4, OED5, OED6):

    # Calculate relative orbital elements
    DA, _, DEX, DEY, DIX, DIY = calculate_ROE(OE1, OE2, OE3, OE4, OE5, OE6, OED1, OED2, OED3, OED4, OED5, OED6)
    
    # Adjust SMA/AS/BS to lower units to minimize singularity
    AS = AS * 1.e3
    BS = BS * 1.e3
    SMA = OE1 * 1.e3
    
    # Compute the coefficients for standard ellipse
    S5 = 1 / AS
    S3 = 1 / BS

    # Calculate ROE phase and terms
    PHI = arctan2(DEY, DEX)
    THETA = arctan2(DIY, DIX)
    DE = sqrt(DEX**2 + DEY**2)
    DI = sqrt(DIX**2 + DIY**2)
    sig_insqr = sqrt(DE**4 + DI**4 - 2 * (DE**2) * (DI**2) * cos(2 * (PHI - THETA)))
    sma_over_sqrt2 = SMA / sqrt(2)

    # Trajectory ellipse parameters
    AG = sma_over_sqrt2 * sqrt(DE**2 + DI**2 + sig_insqr)
    BG = sma_over_sqrt2 * sqrt(DE**2 + DI**2 - sig_insqr)
    PSIG = 1/2 * arctan2(2 * DE * DI * sin(PHI - THETA), DE**2 - DI**2)
    KG = SMA * DA

    # Correct tilt angle
    PSIG = -PSIG + np.sign(PSIG)*np.pi/2

    # Find ellipse dimensions
    S_MAX = max([AS, BS])
    S_MIN = min([AS, BS])
    G_MAX = max([AG, BG])
    G_MIN = min([AG, BG])

    # Check NCA
    if not np.abs(KG) - BS > 0 and not (G_MAX > S_MAX and G_MIN > S_MIN):
        return False
    
    # Compute the coefficients for general ellipse
    T5, T4, T3, T2, T1, T0 = envlpparam2coeffs(AG, BG, PSIG, KG)

    # Calculate H matrix components
    H5 = T5 / S5**2
    H4 = T4 / (S5*S3)
    H3 = T3 / S3**2
    H2 = T2 / S5 
    H1 = T1 / S3
    H0 = T0

    # Compute coefficients for the polynomial
    A3 = H2 + H1 * 1j
    A2 = 2 * (H5 + H3 + 2 * H0)
    A1 = 3 * (H2 - H1 * 1j)
    A0 = 2 * (H5 - H3 - H4 * 1j)

    # Determine polynomial type and compute roots
    if abs(A3) < 1e-15:  # Quadratic or lower
        if abs(A2) < 1e-15:  # Linear or constant
            return False
        else:  # Quadratic case
            discriminant = A1**2 - 4 * A2 * A0
            sqrt_discriminant = sqrt(discriminant + 0j)
            pz = array([(-A1 + sqrt_discriminant) / (2 * A2), (-A1 - sqrt_discriminant) / (2 * A2)])
    else:  # Cubic case
        
        # Solve cubic equation
        delta_0 = A2**2 - 3 * A3 * A1
        delta_1 = 2 * A2**3 - 9 * A3 * A2 * A1 + 27 * A3**2 * A0

        # Compute cubic root terms
        delta_1_sqr = delta_1**2 - 4 * delta_0**3
        sqrt_delta_1_sqr = sqrt(delta_1_sqr + 0j)
        C = ((delta_1 + sqrt_delta_1_sqr) / 2)**(1/3)
        
        # Cube root of unity
        omega = complex(-0.5, sqrt(3) / 2)  # e^(2πi/3)
        omega_sq = omega**2

        # Compute roots
        z1 = -1 / (3 * A3) * (A2 + C + delta_0 / C)
        z2 = -1 / (3 * A3) * (A2 + omega * C + delta_0 / (omega * C))
        z3 = -1 / (3 * A3) * (A2 + omega_sq * C + delta_0 / (omega_sq * C))    
        pz = array([z1, z2, z3])
        
    # (NCB) Check if at least two roots are within the unit disk
    return True if sum(np.abs(pz) < 1) >= 2 else False