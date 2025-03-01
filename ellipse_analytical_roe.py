''''

Analytical Roots of Ellipse-Ellipse

AUTHOR: Bruce L. Barbour, 2024
        Virginia Tech

This Python script employs analytical solutions to explore the root relationships 
between a keep-out ellipse and a trajectory ellipse by evaluating their parameters.
'''

# =================================================================================== #
# ------------------------------- IMPORT PACKAGES ----------------------------------- #
# =================================================================================== #

import numpy as np
from numba import jit, f8
from numpy import sin, cos, sqrt, array, arctan2

# =================================================================================== #
# ------------------------------ UTILITY FUNCTION ----------------------------------- #
# =================================================================================== #
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
    cos_psi = np.cos(Psi)
    sin_psi = np.sin(Psi)
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

    return np.array([c5, c4, c3, c2, c1, c0])

# =================================================================================== #
# -------------------------------- MAIN FUNCTION ------------------------------------ #
# =================================================================================== #
@jit(f8[:](f8, f8, f8, f8, f8, f8, f8, f8), nopython=True, fastmath=True)
def solve_analytical_intersection(AS, BS, SMA, DA, DEX, DEY, DIX, DIY):
    """
    Calculates the existence of an intersection between a keep-out ellipse and a
    trajectory ellipse described in relative orbital elements (ROE).

    When comparing these two ellipses, the keep-out ellipse simplifies the analysis 
    by eliminating the need for decomposition into principal components, as its axes 
    already align with the coordinate axes. Consequently, the trajectory ellipse must be 
    transformed onto the unit disk to align with the keep-out ellipse. This 
    transformation results in a modified trajectory ellipse that can be evaluated 
    relative to the unit disk.

    The coefficients of the transformed ellipse are mapped onto a quartic complex 
    polynomial, preserving the two-dimensional structure. Notably, these coefficients 
    exhibit a conjugate palindromic symmetry, a property that allows for the 
    application of Cohn's theorem. By directly differentiating the quartic polynomial, 
    the problem reduces to a quasi-cubic polynomial. It is referred to as quasi-cubic 
    because the problem can be further reduced to a quadratic polynomial if the condition 
    (a - c - b*I) = 0 is satisfied. This can occur when g_ellip_param[3] is zero.
    
    The resulting complex quasi-cubic polynomial is expressed as:

    P = 2*(a - c - b*I) + 3*(d - e*I) + 2*(a + c + 2*f) + (d + e*I) = 0

    Here, I represents the imaginary unit, defined as sqrt(-1). To determine whether an 
    intersection exists, AG least two roots of the polynomial must lie within the unit 
    disk (radius 1). Mathematically, this condition can be expressed as:

    sum( |zk - 1| < 0 ) >= 2    (INTERSECTION EXISTS)

    where zk is a root of the complex quasi-cubic polynomial P.

    Args:
        AS, BS:     Semi-major Axis (AS) and Semi-minor Axis (BS) of keep-out ellipse
        SMA:        Chief / Client / Target orbital semi-major axis
        DA:         Deputy / Servicer relative semi-major axis ratio
        DEX, DEY:   Deputy / Servicer relative eccentricity components
        DIX, DIY:   Deputy / Servicer relative inclination components   

    Returns:
        bool:                 Existence of intersection
    """

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
    if np.abs(A3) < 1e-15:  # Quadratic or lower
        if np.abs(A2) < 1e-15:  # Linear or constant
            if np.abs(A1) < 1e-15:  # Infinite solutions
                return np.array([1.])
            pz = array([-A0 / A1])  # Linear root
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
        
    # Check if at least two roots are within the unit disk
    return np.array([0.]) if np.sum(np.abs(pz) < 1) >= 2 else np.array([1.])