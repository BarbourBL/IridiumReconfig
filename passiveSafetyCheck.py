
''''

ROE-based Passive Safety Validation Technique

AUTHOR: Bruce L. Barbour, 2023
        Virginia Tech

This Python script encompasses the ROE-based Passive Safety validation technique using
the 2-D projected RC-plane (RIC frame) ellipse of a relative orbit and a keep-out
volume (KOV) centered about a Client spacecraft. These ellipse models are described 
by relative orbital elements (ROE) in the quasi-nonsingular form derived by 
Simone D'Amico, Ph.D.. 

The proposed ROE-based Passive Safety Technique (RPST) is composed of two 
independent sets of necessary and sufficient conditions used to determine
spacecraft passive safety in the context of maintaining integrity of the driving
KOV. Only ONE of these two sets must hold true to satisfy the condition of passive
safety.

    (i) The first set of conditions is a Radial Buffer, specifically a minimum
        radial separation between the KOV and Servicer spacecraft's relative motion.
        In the case of a projected relative orbit in the RC-plane modeled as a 2-D 
        ellipse, the head and/or tail of the ellipse must not connect with the KOV. 
        

        This can be accomplished by ensuring the following condition:

        
            ->  SMA_c*|da| - SMA_c*de > R_KOV

        
        where 'SMA_c' is Client semi-major axis, 'da' is relative semi-major axis,
        'de' is relative eccentricity vector magnitude, and 'R_KOV' is the radius
        of the driving KOV in the radial direction. For completeness, the enclosed 
        bracket '||' is an absolute value bracket.

        
        This condition is necessary AND sufficient. Note that this condition is
        applicable to other forms of relative motion outside of those exhibiting
        an 'elliptical nature'.


    (ii) The second set of conditions is ellipse-centered, in other words, the
         projected RC-plane relative orbit MUST closely resemble a closed ellipse.
         Most often, projected RC-plane orbits will exhibit an elliptical shape
         if the relative eccentricity and inclination vectors are not orthogonal,
         i.e., |atan2(dey, dex) - atan2(diy, dix)| ~= π/2.

        (ii)(a) First condition of the second set guarantees that the 2-D projected
                relative orbit is larger than the KOV. This assumes that a Radial
                Buffer is not observed, thus the imaginary center of the relative 
                orbit is either near or coincident with the Client spacecraft
                position, modeled as the origin in the RIC frame. Ensuring that the
                relative orbit's projected ellipse does not breach the KOV requires
                its semi-major and semi-minor axes extending outside of the KOV 
                radial and cross-track dimensions.

                
                This can be expressed mathematically:

                
                    ->  min(aRC, bRC) > C_KOV
                    ->  max(aRC, bRC) > R_KOV

                
                where 'aRC' and 'bRC' are the semi-major and semi-minor axes of the
                2-D projected ellipse of the relative orbit. The term 'C_KOV' is
                the radius of the driving KOV in cross-track direction. 

        (ii)(b) Second condition of the second set guarantees that no intersection
                exists between the 2-D projected relative orbit and the KOV. This also
                extends to points of contact and/or complete overlap. Compared to the
                first condition in the second set, this condition has higher
                complexity by involving root solver or root behavior estimation
                methods. Therefore, modeling the 2-D projected ellipses of the
                relative orbit and KOV will need to be performed, which the Python
                script 'rpstUtilities.py' provides the necessary pieces.

                Multiple root solver and root behavior estimation methods can be
                utilized to check for intersection. The supplemental Python script
                supplies the depressed quartic function, quartic discriminant, and
                depressed quartic coefficients needed for most, if not all, of the
                methods.

                The methods that will be written as part of the passive safety
                validation algorithm include:

                (1) Nature of Quartic Roots (E.L. Rees)
                (2) Newton-Raphson Method
                (3) Sturm's Theorem
                (4) NumPy PolyRoots (Python library)
                (5) SciPy Optimize f-Solve (Python library)
                (6) Polynomial Conic Intersrction Test (Fitzgerald)

        Both of these conditions MUST be satisfied to be necessary AND sufficient. It
        is possible to have the first condition of the second set be true, but the 
        second of the set to be false (and vise-versa). The reason is that a centered 
        radial separation can exist in the projected RC-plane where a tilt angle can 
        induce ellipse intersection. However, it can be said that having the first 
        condition of the set is definitive since the geometric size of relative orbit 
        being smaller than the KOV, by definition, directly violates passive safety.

Error corrections are introduced to account for floating point and round-off errors in
computation. Additionally, negligible values (i.e., values very close to zero) are
carefully considered by implementing a hard-coded zero-offset via heuristic approach.
'''

# =================================================================================== #
# ------------------------------- IMPORT PACKAGES ----------------------------------- #
# =================================================================================== #

from __future__ import annotations
import time
import numpy as np
import natureofquarticroots as nqr
import brentmethod as bm
import sturmtheorem as st
import polyroots as mnr
import polynomialsolver as ps
import pci
import scipysolver as sci
import parameterization as pts
import householder as hh
import gradient_descent as ecgd
import quart_closedform as qcf
import ellipse_analytical as ea

# =================================================================================== #
# --------------------------- FIRST SET OF CONDITIONS ------------------------------- #
# -------------------------- R A D I A L   B U F F E R ------------------------------ #
# =================================================================================== #

def firstcond_ROE_radialbuffer(
                                ac  : int | float, 
                                roe : np.ndarray,
                                kov : np.ndarray
                              ) -> bool:
    """
    Computes the radial separation of the 2-D ellipses of the Servicer's relative 
    orbit (defined by ROEs) and KOV (defined in RIC-frame) projected in the RC-plane. 
    Then the function checks if the ellipses satisfy the minimum separation condition.

    Args:
        ac (int or float):  Client semi-major axis, in km
        roe (numpy.array):  Dimensionless, quasi-nonsingular ROEs of Servicer
        kov (numpy.array):  RIC-frame radii of driving KOV, in km

    Returns:
        boolean:            Result of the radial buffer check with 'True' as satisfied 
                            condition
    """
    
    # Initially assume a violation
    output = False

    # Radius of KOV in the aligned-radial direction
    R_KOV = kov[0]

    # ROE terms used in check
    da = roe[0]
    de = np.linalg.norm(roe[2:3])

    # Radial buffer condition
    if (np.abs(ac*da) - ac*de > R_KOV):
        output = True
    
    return output

# =================================================================================== #
# --------------------------- SECOND SET OF CONDITIONS ------------------------------ #
# ---------------------- E L L I P S E   D I M E N S I O N S ------------------------ #
# =================================================================================== #

def secondcond_ROE_ellipsedim(
                                ellip_char  : np.ndarray, 
                                kov         : np.ndarray
                             )  -> bool:
    """
    Computes the radial separation of the 2-D ellipses of the Servicer's relative 
    orbit (defined by ROEs) and KOV (defined in RIC-frame) projected in the RC-plane. 
    Then the function checks if the ellipses satisfy the minimum separation condition.

    Args:
        ellip_char (numpy.array):   Characteristics of the RC-plane projected ellipse of
                                    the Servicer's relative orbit
        kov (numpy.array):          RIC-frame radii of driving KOV, in km

    Returns:
        boolean:                    Result of the ellipse dimensional check with 'True' as 
                                    satisfied condition
    """
    
    # Initially assume a violation
    output = False

    # Maximum and minimum radii of projected RC-plane KOV
    kov_max = np.max([kov[0], kov[2]])
    kov_min = np.min([kov[0], kov[2]])

    # Semi-axes of Servicer's ellipse model
    sig_max = ellip_char[0] # Semi-major
    sig_min = ellip_char[1] # Semi-minor

    # Ellipse dimensional condition
    if (sig_max > kov_max and sig_min > kov_min):
        output = True
    
    return output

# =================================================================================== #
# --------------------------- SECOND SET OF CONDITIONS ------------------------------ #
# ------------------- E L L I P S E   I N T E R S E C T I O N ----------------------- #
# =================================================================================== #

def secondcond_ROE_ellipseintersec(
                                    quart_char   : list, 
                                    pf_and_zof   : np.ndarray | None, 
                                    test_to_run  : str
                                   )    -> bool:
    """
    Computes the radial separation of the 2-D ellipses of the Servicer's relative 
    orbit (defined by ROEs) and KOV (defined in RIC-frame) projected in the RC-plane. 
    Then the function checks if the ellipses satisfy the minimum separation condition.

    Args:
        quart_char (list):                  Polynomial expressions for 2-D projected relative 
                                            motion and keep-out volume ellipses, Depressed quartic 
                                            expression evaluated at parameter inputs, quartic 
                                            discriminant, and depressed quartic coefficients as 
                                            functions of projected RC-plane ellipse characteristics

                                            ..OR..

                                            12 ellipse equation coefficients that describe the ellipse
                                            characteristics of the projected keep-out volume and
                                            relative orbit onto the RC-plane
                                            of the KOV and relative orbit
        pf_and_zof (numpy.array):           Max. numerical precision factor and zero offset
        test_to_run (string):               Name of the root-finding/root-estimating method

    Returns:
        boolean:                            Result of the ellipse dimensional check with 'True' 
                                            as satisfied condition
    """
    
    # Nature of Quartic Roots (primary)
    if test_to_run == "NQR":
        return nqr.evaluate_nature_of_quartic_roots(quart_char=quart_char[3:], pf_and_zof=pf_and_zof)
    
    # Brent's Method/Bisection Method
    elif test_to_run == "BM":   
        return bm.evaluate_brents_method(quartf=quart_char[2], int=np.array([-1e100, 1e100]), bisect_only=True, max_iter=100)
    
    # *UPDATED - Sturm's Theorem
    elif test_to_run == "ST":   
        return st.evaluate_sturms_theorem(expr=quart_char[-3], int=np.array([-10, 10]), check_repeated=False)
    
    # *UPDATED - Modified NumPy Root Solver
    elif test_to_run == "MNR":  
        return mnr.modified_roots_method(poly=quart_char[-3], zof=pf_and_zof[1])
    
    # *UPDATED - Parameterization, Theta Sweep (provides full alg. solution)
    elif test_to_run == "PTS":
        return pts.ellipse_parameterization(el_rm=quart_char[3], el_kv=quart_char[4], num_steps=pf_and_zof[0])
    
    # *UPDATED - Halley's Method
    elif test_to_run == "HM":
        return hh.find_root(coef=quart_char[0], x0_g=None, method="halley", bracket=None)
    
    # *UPDATED - Multi-Start Conjugate Gradient
    elif test_to_run == "MSCG":
        return ecgd.mscg(el_rm=quart_char[3], el_kv=quart_char[4], x0_g=None)
    
    # *UPDATED - Closed-Form Equations
    elif test_to_run == "CF":
        return qcf.quartic_closedform(quart_coef=quart_char[0])
    
    # *UPDATED - Symbolic solver
    elif test_to_run == "PS":   
        return ps.evaluate_polynomial_solver(expr1=quart_char[-2], expr2=quart_char[-1])
    
    # Polynomial Conic Intersection (Numpy Roots)
    elif test_to_run == "PCI":
        return pci.evaluate_pci(ellip_coef1=quart_char[2], ellip_coef2=quart_char[1], simplify=False, use_jt=False, eps=None)
    
    # Polynomial Conic Intersection (Numpy Roots - Simplified)
    elif test_to_run == "PCIS":
        return pci.evaluate_pci(ellip_coef1=quart_char[2], ellip_coef2=quart_char[1], simplify=True, use_jt=False, eps=None)
    
    # Polynomial Conic Intersection (Jenkins-Traub)
    elif test_to_run == "PCI-JT":
        return pci.evaluate_pci(ellip_coef1=quart_char[2], ellip_coef2=quart_char[1], simplify=False, use_jt=True, eps=pf_and_zof[1])
    
    # Polynomial Conic Intersection (Jenkins-Traub - Simplified)
    elif test_to_run == "PCIS-JT":
        return pci.evaluate_pci(ellip_coef1=quart_char[2], ellip_coef2=quart_char[1], simplify=True, use_jt=True, eps=pf_and_zof[1])
    
    # SciPy fsolve
    elif test_to_run == "MPM":
        return sci.evaluate_MPM(coef1=quart_char[2], coef2=quart_char[1], init=None)
    
    # Analytical
    elif test_to_run == "EA":
        t0 = time.perf_counter()
        sol = ea.solve_analytical_intersection(AS=quart_char[4][2]*1e3, BS=quart_char[4][0]*1e3, AG=quart_char[3][0]*1e3, BG=quart_char[3][1]*1e3, PSIG=quart_char[3][2], KG=quart_char[3][3]*1e3)
        tf = time.perf_counter()
        return [bool(sol[0]), "Real roots found." if bool(sol[0]) else "No roots found.", tf-t0, quart_char[4], quart_char[3]]

# =================================================================================== #
# -------------------------------- MAIN FUNCTION ------------------------------------ #
# =================================================================================== #

def ROE_Passive_Safety(
                        ac              : int | float, 
                        roe             : np.ndarray | None, 
                        kov             : np.ndarray,
                        ellip_char      : np.ndarray,
                        quart_char      : list, 
                        pf_and_zof      : np.ndarray,
                        test_to_run     : str,
                        force_ellipse   : bool = False
                       )    -> list[bool, str]:
    """
    Evaluates the ROE-based Passive Safety technique using the provided quantities of the Client/Servicer
    relative dynamics and keep-out zones.

    Args:
        ac (int or float):                  Client semi-major axis, in km
        roe (numpy.array):                  Dimensionless, quasi-nonsingular ROEs of Servicer
        kov (numpy.array):                  RIC-frame radii of driving KOV, in km
        ellip_char (numpy.array):           Characteristics of the RC-plane projected ellipse of
                                            the Servicer's relative orbit
        quart_char (list):                  Symbolic polynomial expressions for 2-D projected
                                            relative motion and keep-out volume ellipses, symbolic 
                                            depressed quartic expression, quartic discriminant, and 
                                            depressed quartic coefficients as functions of projected 
                                            RC-plane ellipse characteristics of the KOV and relative orbit
        pf_and_zof (numpy.array):           Max. numerical precision factor and zero-offset 
        test_to_run (string):               Name of the root-finding/root-estimating method
        force_ellipse (bool):               Whether to only use the ellipse conditions. Default is False.                         

    Returns:
        list (boolean and string):  Boolean for ROE-based Passive Safety technique results,
                                    with 'True' as being passively safe

                                    String that detailts the set of necessary and sufficient
                                    conditions used to satisfy passive safety. If ellipse-centered
                                    conditions are used, then information on the intersection test
                                    used and reason for pass/failure are also given.
    """

    # Start timer
    t0 = time.perf_counter_ns()

    # Check if not forced ellipse then use full algorithm
    if not force_ellipse:

        # Checks for first-set of n&s conditions
        if firstcond_ROE_radialbuffer(ac, roe, kov):
            return [True, "SAFE. Radial buffer = SATISFIED!", None, (time.perf_counter_ns()-t0)*1e-9]

        # Checks for second-set of n&s conditions
        elif secondcond_ROE_ellipsedim(ellip_char, kov):
            intersect_test = secondcond_ROE_ellipseintersec(quart_char, pf_and_zof, test_to_run)
            if not intersect_test[0]:
                return [True, "SAFE. Ellipse dim = SATISFIED. Intersection test = SATISFIED: "+test_to_run, intersect_test[2], (time.perf_counter_ns()-t0)*1e-9]
            else:
                return [False, "NOT SAFE. Ellipse dim = SATISFIED. Intersection test = NOT SATISFIED: "+test_to_run, intersect_test[2], (time.perf_counter_ns()-t0)*1e-9]
        else:
            return [False, "NOT SAFE. Ellipse dim = NOT SATISFIED!", None, (time.perf_counter_ns()-t0)*1e-9]
    
    else:

        # Checks for only ellipse conditions
        if secondcond_ROE_ellipsedim(ellip_char, kov):
            intersect_test = secondcond_ROE_ellipseintersec(quart_char, pf_and_zof, test_to_run)
            if not intersect_test[0]:
                return [True, "SAFE. Ellipse dim = SATISFIED. Intersection test = SATISFIED: "+test_to_run, intersect_test[2], (time.perf_counter_ns()-t0)*1e-9]
            else:
                return [False, "NOT SAFE. Ellipse dim = SATISFIED. Intersection test = NOT SATISFIED: "+test_to_run, intersect_test[2], (time.perf_counter_ns()-t0)*1e-9]
        else:
            return [False, "NOT SAFE. Ellipse dim = NOT SATISFIED!", None, (time.perf_counter_ns()-t0)*1e-9]