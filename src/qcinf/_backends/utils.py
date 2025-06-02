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
