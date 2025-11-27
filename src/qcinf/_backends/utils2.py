import numpy as np
from numba import njit

@njit(cache=True)
def _qcp_rmsd_kernel(P, Q):
    """
    QCP-style minimal RMSD between two N×3 point sets using
    Horn's quaternion setup + power iteration to get λ_max.

    Parameters
    ----------
    P, Q : (N, 3) float64 arrays
        Assumed same shape, 2D, and contiguous.

    Returns
    -------
    rmsd : float64
    """
    N = P.shape[0]

    # --- 1) Centroids ---
    cP0 = 0.0; cP1 = 0.0; cP2 = 0.0
    cQ0 = 0.0; cQ1 = 0.0; cQ2 = 0.0

    for i in range(N):
        cP0 += P[i, 0]; cP1 += P[i, 1]; cP2 += P[i, 2]
        cQ0 += Q[i, 0]; cQ1 += Q[i, 1]; cQ2 += Q[i, 2]

    invN = 1.0 / N
    cP0 *= invN; cP1 *= invN; cP2 *= invN
    cQ0 *= invN; cQ1 *= invN; cQ2 *= invN

    # --- 2) Build covariance M and norms Gp, Gq ---
    # M = X^T Y  (3×3), where X = P - cP, Y = Q - cQ
    Sxx = 0.0; Sxy = 0.0; Sxz = 0.0
    Syx = 0.0; Syy = 0.0; Syz = 0.0
    Szx = 0.0; Szy = 0.0; Szz = 0.0

    Gp = 0.0
    Gq = 0.0

    for i in range(N):
        px0 = P[i, 0] - cP0
        px1 = P[i, 1] - cP1
        px2 = P[i, 2] - cP2

        qx0 = Q[i, 0] - cQ0
        qx1 = Q[i, 1] - cQ1
        qx2 = Q[i, 2] - cQ2

        # Accumulate norms
        Gp += px0*px0 + px1*px1 + px2*px2
        Gq += qx0*qx0 + qx1*qx1 + qx2*qx2

        # Covariance terms
        Sxx += px0*qx0; Sxy += px0*qx1; Sxz += px0*qx2
        Syx += px1*qx0; Syy += px1*qx1; Syz += px1*qx2
        Szx += px2*qx0; Szy += px2*qx1; Szz += px2*qx2

    # --- 3) Build Horn/Theobald 4×4 K matrix in scalar form ---
    trace = Sxx + Syy + Szz

    K00 = trace
    K01 = Syz - Szy
    K02 = Szx - Sxz
    K03 = Sxy - Syx

    K11 = Sxx - Syy - Szz
    K12 = Sxy + Syx
    K13 = Szx + Sxz

    K22 = -Sxx + Syy - Szz
    K23 = Syz + Szy

    K33 = -Sxx - Syy + Szz

    # Symmetric entries
    K10 = K01
    K20 = K02
    K21 = K12
    K30 = K03
    K31 = K13
    K32 = K23

    # --- 4) Power iteration to get largest eigenvalue λ_max ---
    # Initial vector
    v0 = 1.0
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0

    lam = 0.0
    lam_prev = 0.0
    max_iter = 50
    tol = 1e-12

    for it in range(max_iter):
        # w = K @ v
        w0 = K00*v0 + K01*v1 + K02*v2 + K03*v3
        w1 = K10*v0 + K11*v1 + K12*v2 + K13*v3
        w2 = K20*v0 + K21*v1 + K22*v2 + K23*v3
        w3 = K30*v0 + K31*v1 + K32*v2 + K33*v3

        # norm(w)
        nrm = (w0*w0 + w1*w1 + w2*w2 + w3*w3)**0.5
        if nrm == 0.0:
            # Degenerate case
            lam = 0.0
            break

        # normalize
        v0 = w0 / nrm
        v1 = w1 / nrm
        v2 = w2 / nrm
        v3 = w3 / nrm

        # Rayleigh quotient λ = v^T K v (v is unit)
        Kv0 = K00*v0 + K01*v1 + K02*v2 + K03*v3
        Kv1 = K10*v0 + K11*v1 + K12*v2 + K13*v3
        Kv2 = K20*v0 + K21*v1 + K22*v2 + K23*v3
        Kv3 = K30*v0 + K31*v1 + K32*v2 + K33*v3

        lam = v0*Kv0 + v1*Kv1 + v2*Kv2 + v3*Kv3

        if it > 0 and abs(lam - lam_prev) < tol*abs(lam):
            break
        lam_prev = lam

    # --- 5) RMSD from λ_max ---
    # Theobald: RMSD^2 = (Gp + Gq - 2 λ_max) / N
    num = Gp + Gq - 2.0*lam
    if num < 0.0:
        num = 0.0  # clamp small negatives from FP
    rmsd = (num / N)**0.5
    return rmsd


def qcp_rmsd_numba(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Public QCP RMSD function.

    Parameters
    ----------
    P, Q : (N,3) array-like
        Coordinates.

    Returns
    -------
    float
        Minimum RMSD between P and Q.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    if P.shape != Q.shape:
        raise ValueError("Coordinate arrays must have the same shape.")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Coordinates must be of shape (N, 3).")

    # Ensure contiguous
    if not P.flags.c_contiguous:
        P = np.ascontiguousarray(P)
    if not Q.flags.c_contiguous:
        Q = np.ascontiguousarray(Q)

    return _qcp_rmsd_kernel(P, Q)
