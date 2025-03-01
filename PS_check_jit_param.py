# =================================================================================== #
# ------------------------------- IMPORT PACKAGES ----------------------------------- #
# =================================================================================== #
import numpy as np
from numba import jit, f8
from numpy import sin, cos, sqrt, array, arctan2, pi, real, imag

# =================================================================================== #
# ------------------------------ UTILITY FUNCTION ----------------------------------- #
# =================================================================================== #
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
def ps_param(AS, BS, ellipse_param):

    # Initialize
    z1 = 0.
    z2 = 0.
    z3 = 0.

    # Unpack
    AG, BG, PSIG, KG = ellipse_param
    
    # Adjust SMA/AS/BS to lower units to minimize singularity
    AS = AS * 1.e3
    BS = BS * 1.e3
    AG = AG * 1.e3
    BG = BG * 1.e3
    KG = (KG + 1e-6) * 1.e3
    PSIG += np.radians(0.03)
    
    # Compute the coefficients for standard ellipse
    S5 = 1 / AS
    S3 = 1 / BS
    
    # Check NCA.II
    S_MAX = max([AS, BS])
    S_MIN = min([AS, BS])
    G_MAX = max([AG, BG])
    G_MIN = min([AG, BG])
    if not (G_MAX > S_MAX and G_MIN > S_MIN):
        return array([2.])
    
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
            if abs(A1) < 1e-15:  # Infinite solutions
                return array([2.])
            z1 = array([-A0 / A1])  # Linear root
        else:  # Quadratic case
            discriminant = A1**2 - 4 * A2 * A0
            sqrt_discriminant = sqrt(discriminant + 0j)
            z1 = (-A1 + sqrt_discriminant) / (2 * A2) 
            z2 = (-A1 - sqrt_discriminant) / (2 * A2)
    else:  # Cubic case
        
        # Solve cubic equation
        delta_0 = A2**2 - 3 * A3 * A1
        delta_1 = 2 * A2**3 - 9 * A3 * A2 * A1 + 27 * A3**2 * A0

        # Compute cubic root terms
        delta_1_sqr = delta_1**2 - 4 * delta_0**3
        sqrt_delta_1_sqr = sqrt(delta_1_sqr + 0j)
        C = max(((delta_1 + sqrt_delta_1_sqr) / 2)**(1/3), ((delta_1 - sqrt_delta_1_sqr) / 2)**(1/3))
        C = 1e-6 if np.abs(C) < 1e-6 else C
        
        # Cube root of unity
        omega = complex(-0.5, sqrt(3) / 2)  # e^(2πi/3)
        omega_sq = omega**2

        # Compute roots
        z1 = -1 / (3 * A3) * (A2 + C + delta_0 / C)
        z2 = -1 / (3 * A3) * (A2 + omega * C + delta_0 / (omega * C))
        z3 = -1 / (3 * A3) * (A2 + omega_sq * C + delta_0 / (omega_sq * C))    
        
    # Break done roots
    solution    = np.zeros(12)

    # Root 1
    solution[0] = abs(z1)
    solution[1] = real(z1)
    solution[2] = imag(z1)
    solution[3] = arctan2(imag(z1), real(z1))

    # Root 2
    solution[4] = abs(z2)
    solution[5] = real(z2)
    solution[6] = imag(z2)
    solution[7] = arctan2(imag(z2), real(z2))

    # Root 3
    solution[8] = abs(z3)
    solution[9] = real(z3)
    solution[10] = imag(z3)
    solution[11] = arctan2(imag(z3), real(z3))

    # Return solution
    return solution

# =================================================================================== #
# ------------------------------- MAIN FUNCTION 2 ----------------------------------- #
# =================================================================================== #
def ps_reg(AS, BS, ellipse_param):

    # Unpack
    AG, BG, PSIG, KG = ellipse_param
    
    # Adjust SMA/AS/BS to lower units to minimize singularity
    AS = AS * 1.e3
    BS = BS * 1.e3
    AG = AG * 1.e3
    BG = BG * 1.e3
    KG = (KG + 1e-6) * 1.e3
    PSIG += np.radians(1e-6)
    
    # Compute the coefficients for standard ellipse
    S5 = 1 / AS
    S3 = 1 / BS
    
    # Check NCA.II
    S_MAX = max([AS, BS])
    S_MIN = min([AS, BS])
    G_MAX = max([AG, BG])
    G_MIN = min([AG, BG])
    if not (G_MAX > S_MAX and G_MIN > S_MIN):
        return array([0.])
    
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
            if abs(A1) < 1e-15:  # Infinite solutions
                return array([0.])
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
        C = max(((delta_1 + sqrt_delta_1_sqr) / 2)**(1/3), ((delta_1 - sqrt_delta_1_sqr) / 2)**(1/3))
        C = 1e-6 if np.abs(C) < 1e-6 else C
        
        # Cube root of unity
        omega = complex(-0.5, sqrt(3) / 2)  # e^(2πi/3)
        omega_sq = omega**2

        # Compute roots
        z1 = -1 / (3 * A3) * (A2 + C + delta_0 / C)
        z2 = -1 / (3 * A3) * (A2 + omega * C + delta_0 / (omega * C))
        z3 = -1 / (3 * A3) * (A2 + omega_sq * C + delta_0 / (omega_sq * C))    
        pz = array([z1, z2, z3])
        
    # (NCB) Check if at least two roots are within the unit disk
    return array([1.]) if sum(np.abs(pz) < 1) >= 2 else array([0.])

# k=1
# print(ps_param(k*0.13, k*0.08, np.array([2.501299999999999635e-01,1.000800000000000023e-01,2.356194490192344837e+00,1.387778780781445676e-16])))