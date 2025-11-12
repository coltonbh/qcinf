# from collections import defaultdict
# from dataclasses import dataclass
# import itertools as it
# from typing import Any, Callable, Sequence

# import numpy as np
# import pynauty as pn
# from qcio import Structure
# from sympy.combinatorics.perm_groups import PermutationGroup
# from sympy.combinatorics.permutations import Permutation


def find_pendant_groups_bct(adj):
    B, cuts = build_block_cut_tree(adj)
    pendants = []

    for cut in cuts:
        leaf_blocks = [b for b in B.neighbors(cut) if B.degree(b) == 1]
        for block in leaf_blocks:
            group_atoms = collect_block_atoms(block)
            # extend outwards for linear chains
            group_atoms = extend_linear_pendants(group_atoms, adj)
            pendants.append((cut, group_atoms))
    return pendants


def _target_indices_for_component(
    C: np.ndarray,
    P: np.ndarray,
    fixed: np.ndarray,
    s1: Structure,
    s2: Structure,
) -> np.ndarray:
    """
    Return the indices in s2 that are the *candidate target pool* for component C,
    based on the current mapping P of C's boundary (neighbors outside C).
    """
    C_set = set(map(int, C))
    # 1) boundary in s1: neighbors of C that are not in C
    boundary = set()
    for u in C:
        for v in s1.adjacency_dict[int(u)]:
            if v not in C_set:
                boundary.add(v)

    # We want boundary images; ideally boundary vertices are fixed by now
    # (typical when backbone is fixed before pendants).
    boundary = np.array(sorted(boundary), dtype=int)
    if boundary.size == 0:
        print("FALLBACK! NO BOUNDARY AT ALL")
        # fully detached component (rare); fall back to canonical pool
        return P[C]

    # Must have mapped boundary; if some aren’t fixed yet, you can either:
    # (a) skip them, or (b) require fixed[boundary].all()
    # For hexane sequencing, boundary is fixed.
    boundary_mapped = P[boundary]  # s2 indices

    # 2) Candidate pool = vertices in s2 adjacent to *all* boundary_mapped
    # (intersection of neighbor sets), then filtered by element types of C.
    neigh_sets = []
    for b2 in boundary_mapped:
        neigh_sets.append(set(s2.adjacency_dict[int(b2)]))
    cand = set.intersection(*neigh_sets) if neigh_sets else set()

    # filter by element types to match C's composition
    # Build multiset of symbols for C
    C_symbols = [s1.symbols[int(i)] for i in C]
    # First, a simple filter: candidates with matching symbol set size
    cand = [j for j in cand if s2.symbols[int(j)] in C_symbols]

    # Optional: if C has >=2 atoms, also check induced-connectivity pattern size
    # (e.g., methyl has 3 H all connected to the same carbon only).
    # For typical CH3 pendants, the simple adjacency-to-boundary filter is enough.

    # If cand is larger than len(C): typically you have exactly the k hydrogens
    # around that mapped carbon; if not, you may refine with symbol counts.
    # Stable order:
    cand_sorted = np.array(sorted(cand), dtype=int)

    # Expect |cand_sorted| == |C|
    # If not equal, you can fall back to P[C] or raise for debugging.
    if cand_sorted.size != C.size:
        print("FALLBACK! MISMATCHED CANDIDATE POOL SIZE")
        # Fallback (keeps things running, but you’ll lose optimality)
        return P[C]

    return cand_sorted


# def _pendant_factor(
#     s1: Structure,
# ) -> list[Batch]:
#     """
#     Factor the structure into a backbone + pendant groups for each element.

#     Returns:
#         List of batches, where the first batch is the backbone and the subsequent batch
#             contains the pendant groups factored by their parent atom and element type.
#     """
#     batches = [[], []]
#     # (parent_idx, symbol) -> list of atom indices
#     pendant_groups: dict[tuple[int, str], list[int]] = defaultdict(list)
#     backbone_indices = set(range(len(s1.symbols)))

#     # Identify pendant groups
#     adj = s1.adjacency_dict
#     for atom_idx, nbrs in adj.items():
#         if len(nbrs) == 1:  # Single attached pendant
#             # Formerly by symbol, now just collapse all multiple pendants
#             # pendant_groups[(nbrs[0], s1.symbols[atom_idx])].append(atom_idx)
#             pendant_groups[nbrs[0]].append(atom_idx)

#     # Build batches: pendants first, then backbone. Sorting for deterministic output.
#     for parent_idx, pendant_idxs in pendant_groups.items():
#         if len(pendant_idxs) > 1:  # Multiple atoms in this pendant group
#             component = Component(
#                 parent_idx=parent_idx, attachment_idx=None, indices=sorted(pendant_idxs)
#             )
#             batches[1].append(component)
#             backbone_indices.difference_update(pendant_idxs)
#     batches[0].append(
#         Component(
#             parent_idx=None, attachment_idx=None, indices=sorted(backbone_indices)
#         )
#     )
#     return batches
