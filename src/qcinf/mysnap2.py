from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np
import pynauty as pn
from qcio import Structure
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation

from .utils import compute_rmsd, kabsch

Index = int
Component = list[Index]
Batch = list[Component]


def _list_of_sets(coloring: list[Any]) -> list[set[Any]]:
    """Convert coloring list to pynauty list of sets format.

    >>> _list_of_sets([0,0,0,1,1,2])
    [{0, 1, 2}, {3, 4}, {5}]
    """
    buckets: dict[int, set[int]] = defaultdict(set)
    for vertex_idx, color in enumerate(coloring):
        buckets[color].add(vertex_idx)
    return list(buckets.values())


def _to_pynauty_graph(adj_dict: dict[int : list[int]], coloring: list[Any]) -> pn.Graph:
    """
    Convert a qcio Structure to a pynauty Graph.

    Parameters:
        adj_dict: Adjacency dictionary of the graph, e.g. {0: [1,2], 1: [0,3], ...}.
        coloring: List of color IDs for each vertex, e.g., [0,0,0,3,3,3]
            indicates atoms 0-2 are of one color and atoms 3-5 are a second color.
            Using the structure's symbols array can be used as a default coloring.
    Returns:
        The corresponding pynauty Graph.
    """
    return pn.Graph(
        number_of_vertices=len(adj_dict.keys()),
        adjacency_dict=adj_dict,
        vertex_coloring=_list_of_sets(coloring),
    )


def _component_partition(
    coloring: list[int], fixed: np.ndarray, C: Sequence[int]
) -> list[int]:
    """
    Start from base_colors (e.g., Nauty coloring). Encode:
      - pointwise stabilizer of 'fixed' by promoting to singletons
      - setwise stabilizer of C by splitting each color into (in-C) vs (out-of-C)
    """
    new_coloring = list(coloring)
    next_id = max(new_coloring) + 1

    # Promote fixed to unique colors (singletons)
    for i, is_fixed in enumerate(fixed):
        if is_fixed:
            new_coloring[i] = next_id
            next_id += 1

    # Place component indices into their own color class by existing color
    for idx in C:
        # Magic number to offset in-C vs out-of-C
        new_coloring[idx] = new_coloring[idx] - 1000000
    return new_coloring


def _map_from_canonical_labels(
    can1: list[int],
    can2: list[int],
) -> list[int]:
    """Generate a map of indices from G1 -> G2 from canonical labels."""
    inv2 = [0] * len(can2)  # label -> original index in s2
    for i, label in enumerate(can2):
        inv2[label] = i
    return [inv2[lab1] for lab1 in can1]


def _restrict_global_generators_to_C(
    gens_global: list[list[int]],
    C: list[int],
) -> list[Permutation]:
    """
    Restrict each global generator (array form on 0..n-1) to a k-point
    Permutation on C, deduplicated.
    Assumes gens_global all preserve C setwise (enforced by the partition).
    """
    g_to_l = {v: i for i, v in enumerate(C)}  # global index -> local index
    seen: set[tuple[int, ...]] = set()
    local: list[Permutation] = []

    for g in gens_global:
        local_map = tuple(g_to_l[g[v]] for v in C)
        if local_map not in seen:
            seen.add(local_map)
            local.append(Permutation(local_map))
    if not local:  # Empty generators (trivial group)
        local.append(Permutation(list(range(len(C)))))  # identity
    return local


def _setwise_stabilizer_on_C(
    adj_dict: dict[int, list[int]],
    base_colors: list[int],
    fixed: np.ndarray,
    C: Sequence[int],
) -> PermutationGroup:
    """(Aut(G)_(fixed))_{C}↾C as a local SymPy group on |C| points.

    This computes the setwise stabilizer of C within the pointwise stabilizer of
    singletons restricted to the action on C.

    Parameters:
        adj_dict: The adjacency dictionary of the structure graph.
        coloring: The vertex coloring (partition) for the whole structure.
        P_opt: Already fixed part of the optimal permutation as a list with None for
            unfixed indices.
        component: The list of indices in the current component.
    """
    # Fix already assigned indices and stabilize component setwise
    colors = _component_partition(base_colors, fixed, C)
    G_c = _to_pynauty_graph(adj_dict, colors)
    gens, *_ = pn.autgrp(G_c)
    # Restrict to C
    local_gens = _restrict_global_generators_to_C(gens, C)
    return PermutationGroup(*local_gens)


def _align_known(
    A: np.ndarray, B: np.ndarray, P: np.ndarray, fixed: np.ndarray
) -> np.ndarray:
    """Return a copy of A aligned to B using pairs (i -> P[i]) for i with fixed[i]."""
    S = np.flatnonzero(fixed)
    idx2 = P[S]
    R, cA, cB = kabsch(A[S], B[idx2])
    return (A - cA) @ R.T + cB  # align all of A


def _pendant_factor(
    s1: Structure,
) -> list[Batch]:
    """
    Factor the structure into a backbone + pendant groups for each element.

    Returns:
        List of batches, where the first batch is the backbone and the subsequent batch
            contains the pendant groups factored by their parent atom and element type.
    """
    batches = [[], []]
    # (parent_idx, symbol) -> list of atom indices
    pendant_groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    backbone_indices = set(range(len(s1.symbols)))

    # Identify pendant groups
    adj = s1.adjacency_dict
    for atom_idx, nbrs in adj.items():
        if len(nbrs) == 1:  # Possible duplicated pendant
            pendant_groups[(nbrs[0], s1.symbols[atom_idx])].append(atom_idx)

    # Build batches: pendants first, then backbone. Sorting for deterministic output.
    for group in pendant_groups.values():
        if len(group) > 1:  # Multiple atoms in this pendant group
            batches[1].append(sorted(group))
            backbone_indices.difference_update(group)
    batches[0].append(sorted(backbone_indices))
    return batches


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

def _target_indices_for_component2(
    C: np.ndarray,
    P: np.ndarray,
    fixed: np.ndarray,
    s1: Structure,
    s2: Structure,
) -> np.ndarray:
    """
    Simpler, neighbor-driven target selection for component C.

    - Uses the current partial map P and fixed mask to locate the 'attachment site'
      in s2 (images of C's boundary).
    - Candidates are the intersection of neighbor sets of those attachment vertices.
    - Greedy, de-duplicated assignment with symbol check; falls back to P[u] if needed.
    """
    C = np.asarray(C, dtype=int)
    Cset = set(C.tolist())

    # 1) Boundary of C in s1 (neighbors outside C)
    boundary = []
    for u in C:
        for v in s1.adjacency_dict[int(u)]:
            if v not in Cset:
                boundary.append(int(v))
    boundary = np.unique(boundary)

    # 2) If no boundary (detached subgraph), we just fall back to canonical pool P[C]
    if boundary.size == 0:
        return P[C]

    # 3) Map boundary via current P to s2, then intersect neighbor sets in s2
    boundary_img = P[boundary]  # s2 indices
    neigh_sets = [set(s2.adjacency_dict[int(b2)]) for b2 in boundary_img]
    attach_pool = set.intersection(*neigh_sets) if neigh_sets else set()

    # 4) Candidate list per u ∈ C: symbol match + in attach_pool
    cand_per_u = []
    for u in C:
        sym_u = s1.symbols[int(u)]
        cand_u = {j for j in attach_pool if s2.symbols[int(j)] == sym_u}
        cand_per_u.append((int(u), cand_u))

    # 5) Greedy, smallest domain first; avoid already-used s2 targets
    #    Start with s2 indices already used by fixed assignments.
    used = set(P[np.flatnonzero(fixed)].tolist())
    # (You can also optionally seed with any P[C’] for components already locked.)

    # Order C by fewest candidates to stabilize greedy assignment
    order = sorted(range(len(C)), key=lambda i: len(cand_per_u[i][1]))
    T = np.empty(len(C), dtype=int)

    for idx in order:
        u, cand = cand_per_u[idx]
        # Prefer unused candidates
        choices = sorted(j for j in cand if j not in used)
        if not choices:
            # Fallback: keep canonical image for u (keeps things moving)
            j = int(P[int(u)])
            # If even that is already used (rare), pick any unused neighbor of the attachment pool
            if j in used:
                pool_choices = sorted(x for x in attach_pool if x not in used)
                if pool_choices:
                    j = pool_choices[0]
        # Record
        T[np.where(C == u)[0][0]] = j
        used.add(j)

    return T


# def symmetry_aware_factor(s: Structure, depth=∞, force_depth=False):
#     G = graph_from_structure(s)
#     expansion_map = {}     # supernode -> original vertices
#     bundles = []           # list[Component] for fitting

#     for d in range(depth):
#         branches = find_candidate_branches(G)         # e.g., BC-tree leaves at articulations
#         marks = []
#         for a, H in branches:                         # anchor a, branch subgraph H
#             if not force_depth:
#                 if wl_gate_says_asymmetric(G, a, H):
#                     continue
#                 if not anchor_stabilizer_nontrivial(G, a, H):  # one nauty call on H′
#                     continue
#             else:
#                 if is_small_hetero_tail(H) and not has_duplicate_siblings(G, a, H):
#                     continue                          # guardrail

#             marks.append((a, H))                      # collapse

#         if not marks:
#             break

#         # collapse marked branches → supernodes, record expansion_map
#         G, new_supernodes = collapse(G, marks, expansion_map)

#         # within each anchor, bundle identical branches by a branch hash
#         bundles.extend(bundle_identical_siblings(new_supernodes))

#     return G, bundles, expansion_map


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



def snap_rmsd(
    s1: Structure,
    s2: Structure,
    align: bool = True,
    factor: str | Callable[[Any], list[Batch]] | None = "pendant",
    backend: str = "pynauty",
    realign_per_component: bool = False,
) -> tuple[float, list[int]]:
    """
    Compute snapRMSD(s1→s2). Returns (rmsd, P) where P optimally maps s1 indices to s2 indices.

    Strategy:
      - Build graphs; require isomorphism.
      - Initialize global map P from canonical labels (M).
      - For each batch:
          * For each component C:
              · Build local setwise stabilizer (given 'fixed')
              · Score candidates WITHOUT mutating P (zero-copy indices)
              · Lock-in winner: P[C] = base_C[best_perm]
          * After batch, optionally align entire G1 to G2 using (P, fixed)
      - Final RMSD on aligned coordinates (or raw if align=False).

    Parameters:
        s1: First structure.
        s2: Second structure.
        align: Whether to align the structures before computing RMSD. Default is True.
            Generally one should only set this to False if the structures should be
            compared in their absolute orientations, e.g., from a crystal structure
            or docking pose.
        factor: Either a string specifying the factoring method, a callable that
            performs the factoring, or a batches object that factors the structure.
            Default is "pendant". Factoring should return list[Batch] where each Batch
            is a list of components and each component is a list of atom indices
            corresponding to s1, e.g., [[ [0,1,2], [3,4] ], [ [5,6], [7] ]].
        backend: The backend to use for automorphism computations. Default is "pynauty".
        realign_per_component: Whether to realign the structures after each factored component
            is processed. Default is False meaning that realignment is only done after
            each batch, except for the first batch which must align for each proposed component
            to establish a common reference frame.

    Returns:
        (float, P_opt): The computed snapRMSD value and the optimal permutation as a
            list of indices for s1 onto s2.
    """
    # --- 1. Build graphs, canonical labels map, compute automorphism group -----------
    # NOTE: Currently running nauty 3x on G1 2x on G2: once for generators, once for canon label,
    #  once for isomorphism check. Can optimize this.

    # --- Graphs & isomorphism check
    A_adj = s1.adjacency_dict  # compute once since used multiple times
    A = _to_pynauty_graph(A_adj, s1.symbols)
    B = _to_pynauty_graph(s2.adjacency_dict, s2.symbols)

    if not pn.isomorphic(A, B):
        raise ValueError("Structures not isomorphic. Same connectivity required.")

    # --- Canonical labels & initial map
    P = np.array(_map_from_canonical_labels(pn.canon_label(A), pn.canon_label(B)))
    F = np.zeros(len(s1.symbols), dtype=bool)  # track fixed indices

    # --- Use Nauty's coloring as base partition
    A_gens, A_mantisa, A_exponent, A_coloring, A_num_orbits = pn.autgrp(A)

    A_geom = s1.geometry.copy()
    B_geom = s2.geometry.copy()

    # --- 2. Factor batches ----------------------------------------------------------
    if isinstance(factor, str):
        if factor.lower() == "pendant":
            batches = _pendant_factor(s1)
        else:
            raise ValueError(f"Unknown factoring method '{factor}'.")
    elif factor is None:
        batches = [[list(range(len(s1.symbols)))]]
    else:
        batches = factor(s1)  # expected: list[Batch]

    # --- 3. Main control loop -----------------------------------------------------
    for b_idx, batch in enumerate(batches):
        for c_idx, C in enumerate(map(np.array, batch)):
            best_rmsd = float("inf")
            best_perm: list[int] | None = None

            # Precompute fixed sets for this component
            S_fixed = np.flatnonzero(F)
            idx2_fixed = P[S_fixed]

            # Local action group on C given current 'fixed'
            Ac_local = _setwise_stabilizer_on_C(A_adj, A_coloring, F, C)

            for perm in Ac_local.generate_schreier_sims(af=True):
                # s2 target pool for C
                idx2_C = P[C][perm]
                # T = _target_indices_for_component(C, P, fixed, s1, s2)
                # T = _target_indices_for_component2(C, P, fixed, s1, s2)
                # idx2_C = T[perm]
                

                if align and (realign_per_component or (b_idx == 0 and c_idx == 0)):
                    # score on fixed ∪ C, with a per-candidate temporary alignment
                    S = np.concatenate([S_fixed, C])
                    idx2 = np.concatenate([idx2_fixed, idx2_C])
                    A_S = A_geom[S]
                    B_S = B_geom[idx2]

                    # do NOT mutate G1_geom here; align the subset temporarily
                    R, cA_S, cB_S = kabsch(A_S, B_S)
                    A_S_aligned = (A_S - cA_S) @ R.T + cB_S
                    score = compute_rmsd(A_S_aligned, B_S, align=False)

                else:
                    # component-only scoring, no per-candidate alignment
                    A_C = A_geom[C]
                    B_C = B_geom[idx2_C]
                    score = compute_rmsd(A_C, B_C, align=False)

                if score < best_rmsd:
                    best_rmsd = score
                    best_perm = perm

            # Lock-in winner permutation on C
            # P[C] = T[best_perm]
            P[C] = P[C][best_perm]
            F[C] = True

            # Align first component in batch to establish common frame
            if align and b_idx == 0 and c_idx == 0:
                A_geom = _align_known(A_geom, B_geom, P, F)

        # Batch-level global realignment; not realign to avoid duplicated effort
        if align and not realign_per_component:
            A_geom = _align_known(A_geom, B_geom, P, F)

    assert np.all(F), "Not all indices got fixed!."  # Sanity check while debugging
    # Align false because already aligned in the loop if needed
    final_rmsd = compute_rmsd(A_geom, B_geom[P], align=False)
    return float(final_rmsd), P.tolist()


if __name__ == "__main__":
    from time import time

    from qcio import Structure
    from spyrmsd.rmsd import symmrmsd

    from qcinf import smiles_to_structure
    from qcinf.algorithms.geometry import rmsd

    # smiles = "FC(F)(F)C1=CC(=CC(NC(=O)NC2=CC(=CC(=C2)C(F)(F)F)C(F)(F)F)=C1)C(F)(F)F"
    smiles = "CCCCC"
    s1 = smiles_to_structure(smiles)
    s2 = smiles_to_structure(smiles)

    batches = _pendant_factor(s1)
    print(batches)
    start = time()
    s_rmsd, perm = snap_rmsd(s1, s2, align=True, factor="pendant")
    # s_rmsd, perm = snap_rmsd(s1, s2, align=True, factor=None)
    snap_time = time() - start

    start = time()
    rdkit_rmsd = rmsd(s1, s2, backend="rdkit")
    rdkit_time = time() - start

    start = time()
    spyrmsd_rmsd = symmrmsd(
        s1.geometry,
        s2.geometry,
        s1.symbols,
        s2.symbols,
        s1.adjacency_matrix,
        s2.adjacency_matrix,
        center=True,
        minimize=True,
        cache=False,
    )
    spyrmsd_rmsd_time = time() - start

    raw = compute_rmsd(s1.geometry, s2.geometry, align=False)

    print(f"{s_rmsd:.5f}", perm)
    print(f"{spyrmsd_rmsd:.5f}")
    print(f"{rdkit_rmsd:.5f}")
    print(raw)
    print(f"snapRMSD time: {snap_time:.4f} s")
    print(f"spyrmsd RMSD time: {spyrmsd_rmsd_time:.4f} s")
    print(f"RDKit RMSD time: {rdkit_time:.4f} s")
