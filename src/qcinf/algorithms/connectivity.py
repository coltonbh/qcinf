import numpy as np
import qcconst.periodic_table as pt
from qcio import Structure


def determine_connectivity(
    structure: Structure, *, cov_factor: float = 1.2
) -> list[tuple[int, int, float]]:
    """Determine connectivity using covalent radii.

    Two atoms are considered bonded when their distance is less than or equal to
    `cov_factor * (r_cov(i) + r_cov(j))`. Bond orders are set to 1.0 since this
    routine only infers connectivity. Numpy vectorization is used for speed (~27x faster
    than nested loops for a 40-atom molecule, 55x faster for a 113-atom molecule).

    Args:
        structure: The Structure containing symbols and geometry in Bohr.
        cov_factor: Scaling factor applied to the sum of covalent radii.

    Returns:
        A list of (atom_index_a, atom_index_b, bond_order) tuples, using 0-based
        indices and a bond order of 1.0.
    """
    if cov_factor <= 0:
        raise ValueError("cov_factor must be positive.")

    geom = structure.geometry
    syms = structure.symbols
    n = geom.shape[0]

    # 1. Covalent radii as an array
    radii = np.array([getattr(pt, sym).cov_radius for sym in syms], dtype=float)

    # 2. All (i, j) with i < j
    i_idx, j_idx = np.triu_indices(n, k=1)

    # 3. Pairwise distances for just those pairs
    deltas = geom[i_idx] - geom[j_idx]  # shape (num_pairs, 3)
    dists = np.linalg.norm(deltas, axis=1)  # shape (num_pairs,)

    # Threshold for each pair
    thresholds = cov_factor * (radii[i_idx] + radii[j_idx])

    # 4. Mask pairs that are bonded
    mask = dists <= thresholds
    bonded_i = i_idx[mask]
    bonded_j = j_idx[mask]

    # Build connectivity list with bond order 1.0
    connectivity = [(int(i), int(j), 1.0) for i, j in zip(bonded_i, bonded_j)]
    return connectivity
