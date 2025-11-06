from typing import Any, Callable, Iterable

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
    d: dict[int, set[int]] = {}
    for vertex_idx, color in enumerate(coloring):
        d.setdefault(color, set()).add(vertex_idx)
    return list(d.values())


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
    coloring: list[int], p_opt: list[int | None], component: list[int]
) -> list[int]:
    """
    Return a component-specific coloring (partition) that encodes:
      - pointwise stabilizer of fixed vertices (singletons),
      - setwise stabilizer of 'component' (split colors into in-C vs out-of-C).

    Parameters:
        coloring: The coloring for the whole graph.
        p_opt: The current optimal permutation as a list with None for unfixed indices.
            These will be promoted to singletons.
        component: The list of indices for the component.
    """
    c_partition = list(coloring)  # start with existing orbits
    next_id = max(c_partition) + 1

    # Promote fixed indices to singletons (unique colors)
    for i, p in enumerate(p_opt):
        if p >= 0:
            c_partition[i] = next_id  # unique color for fixed indices
            next_id += 1

    # Place component indices into their own color class by existing color
    c_map: dict[int, int] = {}
    for idx in component:
        if c_partition[idx] not in c_map:
            c_map[c_partition[idx]] = next_id
            next_id += 1
        c_partition[idx] = c_map[c_partition[idx]]
    return c_partition


def _map_from_canonical_labels(
    can1: list[int],
    can2: list[int],
) -> list[int]:
    """Generate a map of indices from G1 -> G2 from canonical labels."""
    inv2 = [0] * len(can1)
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


def _setwise_stabilizer(
    adj_dict: dict[int : list[int]],
    coloring: list[int],
    P_opt: list[int],
    component: list[int],
) -> Iterable[Permutation]:
    """Yield permutations in the setwise stabilizer of C (thus permuting C among itself).

    This computes the setwise stabilizer of C within the pointwise stabilizer of
    singletons restricted to the action on C, i.e., (Aut(G)_(fixed))_{C}↾C.


    Parameters:
        adj_dict: The adjacency dictionary of the structure graph.
        coloring: The vertex coloring (partition) for the whole structure.
        P_opt: Already fixed part of the optimal permutation as a list with None for
            unfixed indices.
        component: The list of indices in the current component.
    """
    # Fix already assigned indices and stabilize component setwise
    color_with_singletons = _component_partition(coloring, P_opt, component)
    G_component = _to_pynauty_graph(adj_dict, color_with_singletons)
    GC_gens, *_ = pn.autgrp(G_component)
    # Restrict to C
    local_gens = _restrict_global_generators_to_C(GC_gens, component)
    G_local = PermutationGroup(*local_gens)
    return G_local


def _align(
    A_coords: np.ndarray,
    B_coords: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Align A_coords to B_coords using known mappings in P."""
    known = P >= 0  # boolean mask for known indices
    if known.sum() >= 3:  # gaurd: Kabsch requires three points
        idx2 = P[known].astype(int)
        A_known = A_coords[known]
        B_known = B_coords[idx2]
        R, c1, c2 = kabsch(A_known, B_known)
        A_aligned = (A_coords - c1) @ R.T + c2
        return A_aligned
    return A_coords


def snap_rmsd(
    s1: Structure,
    s2: Structure,
    align: bool = True,
    factor: str | Callable = "pendant",
    backend: str = "pynauty",
    realign_per_component: bool = False,
) -> tuple[float, list[int]]:
    """
    Compute the snapRMSD between two structures using the specified factoring method.

    Parameters:
        s1: First structure.
        s2: Second structure.
        align: Whether to align the structures before computing RMSD. Default is True.
            Generally one should only set this to False if the structures should be
            compared in their absolute orientations, e.g., from a crystal structure
            or docking pose.
        factor: Either a string specifying the factoring method or a callable that
            performs the factoring. Default is "pendant". Factoring should return a list
            of batches, where each batch is a list of components and each component is
            a list of atom indices corresponding to s1.
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
    G1_adj = s1.adjacency_dict  # compute once since used multiple times
    G1 = _to_pynauty_graph(G1_adj, s1.symbols)
    G2 = _to_pynauty_graph(s2.adjacency_dict, s2.symbols)

    if not pn.isomorphic(G1, G2):
        raise ValueError("Structures not isomorphic. Same connectivity required.")

    M = np.array(_map_from_canonical_labels(pn.canon_label(G1), pn.canon_label(G2)))
    G1_gens, G1_mantisa, G1_exponent, G1_coloring, G1_num_orbits = pn.autgrp(G1)

    G1_geom = s1.geometry.copy()
    G2_geom = s2.geometry.copy()
    P_opt = np.full(len(s1.symbols), -1, dtype=int)  # -1 means unfixed

    # --- 2. Factor batches ----------------------------------------------------------
    # TODO: Implement more factoring methods and a validate batches function
    batches: list[Batch] = [[list(range(len(s1.symbols)))]]
    # batches = [[[1, 5, 6, 7], [0, 2, 3, 4]]]  # hardcoded for ethane

    # --- 3. Main control loop -----------------------------------------------------
    for batch_idx, batch in enumerate(batches):
        for c_idx, C in enumerate(map(np.array, batch)):
            best_rmsd = float("inf")
            best_perm = None
            G1C_geom = G1_geom[C]
            # Restricted subgroup elements that only act on C (fix complement pointwise)
            G1C_local = _setwise_stabilizer(G1_adj, G1_coloring, P_opt, C)
            for perm in G1C_local.generate_schreier_sims():
                # Build mapping P_c: G1 global idx -> G2 global idx
                P_c = M[C[perm.array_form]]

                if align and ((batch_idx == 0 and c_idx == 0) or realign_per_component):
                    # Handle first-component and/or per-component realignment if requested
                    # Build trial mapping over P_trial = fixed ∪ C
                    P_trial = P_opt.copy()
                    P_trial[C] = P_c

                    # Align into a TEMP copy — do not mutate G1_geom here
                    G1_geom_trial = _align(G1_geom, G2_geom, P_trial)

                    known = P_trial >= 0  # boolean mask for known indices
                    idx2 = P_trial[known].astype(int)
                    G1_trial = G1_geom_trial[known]
                    G2_trial = G2_geom[idx2]
                else:
                    # No per-candidate realign; evaluate only on the component
                    G1_trial = G1_geom[C]
                    G2_trial = G2_geom[P_c]

                rmsd = compute_rmsd(G1_trial, G2_trial, align=False)
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_perm = P_c

            # Update P_opt with best found permutation for this component
            for G1_idx, G2_idx in zip(C, best_perm):
                P_opt[G1_idx] = int(G2_idx)

        # After each batch "lock in" the best permutations by a global realignment
        if align and not realign_per_component:
            G1_geom = _align(G1_geom, G2_geom, P_opt)

    # Sanity check while debugging
    assert np.all(P_opt >= 0), "Final permutation incomplete."
    # Align false because already aligned in the loop if needed
    final_rmsd = compute_rmsd(G1_geom, G2_geom[P_opt], align=False)
    return float(final_rmsd), P_opt.tolist()


if __name__ == "__main__":
    from time import time

    from qcio import Structure
    from spyrmsd.rmsd import symmrmsd

    from qcinf import rmsd, smiles_to_structure

    smiles = "FC(F)(F)C1=CC(=CC(NC(=O)NC2=CC(=CC(=C2)C(F)(F)F)C(F)(F)F)=C1)C(F)(F)F"
    # smiles = "CCCCCC"
    s1 = smiles_to_structure(smiles)
    s2 = smiles_to_structure(smiles)

    start = time()
    s_rmsd, perm = snap_rmsd(s1, s2, align=True)
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
    )
    spyrmsd_rmsd_time = time() - start

    raw = compute_rmsd(s1.geometry, s2.geometry, align=False)

    print(s_rmsd, perm)
    print(spyrmsd_rmsd)
    print(rdkit_rmsd)
    print(raw)
    print(f"snapRMSD time: {snap_time:.3f} s")
    print(f"spyrmsd RMSD time: {spyrmsd_rmsd_time:.3f} s")
    print(f"RDKit RMSD time: {rdkit_time:.3f} s")
    
