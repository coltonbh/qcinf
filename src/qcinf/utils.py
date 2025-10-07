import numpy as np
from qcio import Structure


def rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix for a rotation about a given axis by angle in degrees.

    Parameters:
        axis (str): 'x', 'y', or 'z' specifying the rotation axis.
        angle_deg (float): Rotation angle in degrees.

    Returns:
        np.ndarray: 3x3 rotation matrix.
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
