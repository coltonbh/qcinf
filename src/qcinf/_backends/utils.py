"""
Utility functions to support backend operations. Functions
may be lower-level and need not expose a high-level qcio API.
"""

import os
import threading
from contextlib import contextmanager

import numpy as np
from qcio import Structure

# one global lock per process
_STDERR_LOCK = threading.Lock()


@contextmanager
def mute_c_stderr():
    """
    Redirect the C-level `stderr` (fd 2) to /dev/null for the duration
    of the context.  This silences C / C++ libraries like RDKit that
    write directly with `fprintf(stderr, …)` or `std::cerr`.

    Acquires a global lock (to make it thread-safe), redirects fd 2 to /dev/null,
    runs the body, then restores stderr and releases the lock.

    Be aware that if another thread NOT using this context manager writes to
    stderr while this lock is held, it will be lost (written to /dev/null).
    In practice this is rare, just be aware of it.
    """
    with _STDERR_LOCK:
        # Duplicate the original fd so we can restore later
        orig_fd = os.dup(2)

        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), 2)  #  ← fd 2 now points to /dev/null
            yield
        finally:
            os.dup2(orig_fd, 2)  #  Restore real stderr
            os.close(orig_fd)


def rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix for a rotation about a given axis by angle in degrees.

    Parameters:
        axis (str): 'x', 'y', or 'z' specifying the rotation axis.
        angle_deg (float): Rotation angle in degrees.

    Returns:
        np.ndarray: 3x3 rotation matrix.

    Example:
        >>> rotation_matrix('z', 90)
        array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    if axis.lower() == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis.lower() == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis.lower() == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")


def rotate_structure(struct: Structure, axis: str, angle_deg: float) -> Structure:
    """
    Return a new Structure with its coordinates rotated by angle_deg about the given axis.

    Parameters:
        struct (Structure): Input structure with a .geometry attribute (an N x 3 numpy array).
        axis (str): Axis to rotate about ('x', 'y', or 'z').
        angle_deg (float): Rotation angle in degrees.

    Returns:
        Structure: New structure with rotated coordinates.
    """
    R = rotation_matrix(axis, angle_deg)
    # Dump the structure to a dictionary and modify the geometry.
    new_struct = struct.model_dump()
    # Apply rotation: for each coordinate, multiply with the rotation matrix.
    # We use R.T because our coordinates are row vectors.
    new_struct["geometry"] = np.dot(struct.geometry, R.T)
    return Structure.model_validate(new_struct)


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the optimal rotation matrix that aligns P onto Q using the Kabsch algorithm.

    Args:
        P (np.ndarray): An N x 3 array of coordinates (the structure to rotate).
        Q (np.ndarray): An N x 3 array of coordinates (the reference structure).

    Returns:
        R (np.ndarray): The optimal 3x3 rotation matrix.
        centroid_P (np.ndarray): The centroid of P.
        centroid_Q (np.ndarray): The centroid of Q.
    """
    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Center the coordinates
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute covariance matrix
    H = np.dot(P_centered.T, Q_centered)
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Correct for reflection (ensure a proper rotation with det(R)=1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R, centroid_P, centroid_Q


def compute_rmsd(
    coords1: np.ndarray, coords2: np.ndarray, *, align: bool = True
) -> float:
    """
    Compute the RMSD between two sets of coordinates.

    Args:
        coords1 (np.ndarray): An N x 3 array of coordinates.
        coords2 (np.ndarray): An N x 3 array of coordinates.
        align (bool): If True, align coords1 to coords2 using the Kabsch algorithm
            before computing RMSD.

    Returns:
        float: The RMSD value.
    """
    # Ensure the arrays have the same shape
    assert coords1.shape == coords2.shape, "Coordinate arrays must be the same shape."

    if align:
        # Align coords1 to coords2 using the Kabsch algorithm
        R, centroid_P, centroid_Q = kabsch(coords1, coords2)
        # Rotate coords1 and translate to the centroid of coords2
        coords1 = np.dot(coords1 - centroid_P, R.T) + centroid_Q

    # Compute the difference between the two arrays
    diff = coords1 - coords2
    # Sum squared differences for each atom (row) and average over all atoms
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def _permute_to_best_match(
    coords1: np.ndarray, coords2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return coords2 permuted to best match coords1 (minimal RMSD).

    Uses Hungarian on the pair-wise distance matrix *after* a first Kabsch
    superposition pass.
    """
    # 1. Rough Kabsch alignment to reduce distance spread
    R, c1, c2 = kabsch(coords1, coords2)
    coords2_aligned = (coords2 - c2) @ R.T + c1

    # 2. Build cost matrix of squared distances
    d2 = np.sum((coords1[:, None, :] - coords2_aligned[None, :, :]) ** 2, axis=2)

    perm, _ = _hungarian(d2)
    return coords2[perm], coords2_aligned[perm]


def _hungarian(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Minimal-cost assignment using the Hungarian algorithm.

    Parameters
    ----------
    cost : (N, N) ndarray
        Square cost matrix.  `cost[i, j]` is the cost of matching
        row *i* (atom *i* in struct1) to column *j* (atom *j* in struct2).

    Returns
    -------
    assign : ndarray of shape (N,)
        `assign[i] == j` means row *i* is matched to column *j*.
    total_cost : float
        Sum of the costs for the optimal assignment.
    """
    C = cost.copy()
    n = C.shape[0]

    # --- Step 1: subtract row minima
    C -= C.min(axis=1, keepdims=True)
    # --- Step 2: subtract column minima
    C -= C.min(axis=0, keepdims=True)

    # Masks for starred / primed zeros
    star = np.zeros_like(C, dtype=bool)
    prime = np.zeros_like(C, dtype=bool)
    covered_rows = np.zeros(n, dtype=bool)
    covered_cols = np.zeros(n, dtype=bool)

    # Helper to star one zero per row (first pass)
    for r in range(n):
        c = np.where((C[r] == 0) & ~covered_cols)[0]
        if c.size:
            star[r, c[0]] = True
            covered_cols[c[0]] = True
    covered_cols[:] = False  # reset

    def _cover_starred_cols() -> None:
        """Cover every column that contains a starred zero."""
        covered_cols[:] = star.any(axis=0)

    _cover_starred_cols()

    while covered_cols.sum() < n:
        # Step 4: find a non-covered zero
        while True:
            rows, cols = np.where((C == 0) & ~covered_rows[:, None] & ~covered_cols)
            if rows.size == 0:
                # Step 6: add smallest uncovered value to covered rows
                min_uncovered = C[~covered_rows[:, None] & ~covered_cols].min()
                C[~covered_rows[:, None] & ~covered_cols] -= min_uncovered
                C[covered_rows[:, None] & covered_cols] += min_uncovered
                rows, cols = np.where((C == 0) & ~covered_rows[:, None] & ~covered_cols)
            r, c = rows[0], cols[0]
            prime[r, c] = True
            # If there is a starred zero in this row, cover the row and uncover the column
            c_star = np.where(star[r])[0]
            if c_star.size:
                covered_rows[r] = True
                covered_cols[c_star[0]] = False
            else:
                # Step 5: augment path
                path: list[tuple[int, int]] = [(r, c)]
                while True:
                    r_star = np.where(star[:, path[-1][1]])[0]
                    if r_star.size == 0:
                        break
                    path.append((r_star[0], path[-1][1]))
                    c_prime = np.where(prime[path[-1][0]])[0]
                    path.append((path[-1][0], c_prime[0]))
                for rr, cc in path:
                    star[rr, cc] = not star[rr, cc]
                    prime[rr, cc] = False
                prime[:] = False
                covered_rows[:] = False
                _cover_starred_cols()
                break

    assign = star.argmax(axis=1)
    total = cost[np.arange(n), assign].sum()
    return assign, float(total)
