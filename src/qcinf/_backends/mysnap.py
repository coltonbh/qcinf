import itertools as it
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pynauty as pn
from qcio import Structure
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation

from .utils import compute_rmsd, kabsch


def _list_of_sets(coloring: list[Any]) -> list[set[Any]]:
    """Convert coloring list to pynauty list of sets format.

    !!! Important
        The order of the sets in the output list MUST be consistent between graphs.
        For this reason we sort the color keys before constructing the output list.


    >>> _list_of_sets(["H","O","C","H","C","C"])
    [{2, 4, 5}, {0, 3}, {1}]  # In alphabetical order: C, H, O

    >>> _list_of_sets([2,1,0,1,0,0])
    [{2, 4, 5}, {1, 3}, {0}]  # In numerical order: 0, 1, 2
    """
    buckets: dict[int, set[int]] = defaultdict(set)
    for vertex_idx, color in enumerate(coloring):
        buckets[color].add(vertex_idx)
    ordered_colors = sorted(buckets.keys())
    return [buckets[colors] for colors in ordered_colors]


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


def _map_via_canonical_labels(
    A: pn.Graph,
    B: pn.Graph,
) -> list[int]:
    """Generate a mapping of vertices from A -> B using canonical labels.
    The final mapping P satisfies P[i_A] = i_B.

    Note:
        pynauty's canonical labels map canon -> original index, meaning that
        canX[i_canon] = i_orig.
    """
    canA, canB = pn.canon_label(A), pn.canon_label(B)
    inv1 = [0] * len(canA)  # inv(can1): orig1 -> canon
    for i_canon, i_orig1 in enumerate(canA):
        inv1[i_orig1] = i_canon
    # P[i_orig1] = can2[canon_of_i_orig1]
    return [canB[inv1[i_orig1]] for i_orig1 in range(len(canA))]


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
        fixed: Boolean array marking fixed vertices.
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


@dataclass
class Component:
    """A component with optional parent and attachment indices.

    Attributes:
        parent_idx: The index of the parent component this component is attached to. No
            parent_idx means this is a backbone component.
        pool: Whether this component is a pool pendant (True) or non-pool pendant (False).
        idxs: The indices of the atoms in this component.
    """

    parent_idx: int | None
    pool: bool
    idxs: np.ndarray


Batch = list[Component]


def component_permutations(
    adj_dict: dict[int, list[int]],
    base_colors: list[int],
    fixed: np.ndarray,
    component: Component,
) -> Iterable[list[int]]:
    """Aut(G)_{fixed}↾C as a local SymPy group on |C| points.

    This computes the pointwise stabilizer of the fixed vertices restricted to
    the action on C.

    Parameters:
        adj_dict: The adjacency dictionary of the structure graph.
        coloring: The vertex coloring (partition) for the whole structure.
        fixed: Boolean array marking fixed vertices.
        component: The list of indices in the current component.

    Returns:
        An iterable of permutations (as lists) acting on the component indices
        reindex to 0..|C|-1.
    """
    if component.pool:  # Direct permutations by element type
        # Group by color (proxy for element types)
        groups = defaultdict(list)
        for i, c_idx in enumerate(component.idxs):
            groups[base_colors[c_idx]].append(i)
        # e.g., -CH2F -> [[0,1],[2]] -> permutations of [0,1] cross [2]
        # which yields [[0,1,2], [1,0,2]] for the full component
        return (
            list(it.chain.from_iterable(combo))
            for combo in it.product(
                *(it.permutations(idxs) for idxs in groups.values())
            )
        )
    else:  # Backbone or non-pool pendant: full setwise stabilizer
        return _setwise_stabilizer_on_C(
            adj_dict, base_colors, fixed, component.idxs
        ).generate_schreier_sims(af=True)


def _pendant_factor(s1: Structure, depth: int = 1) -> dict[int | None, Component]:
    """
    Factor the structure into a backbone + pendant groups for each element.

    For now just one layer and using a dictionary to group pendants by parent atom.

    Parameters:
        s1: The structure to factor.
        depth: The depth of pendant groups to consider (default 1). 0 means no pendant
            factoring (the whole structure will be considered backbone). 1 means singly
            attached pendants with siblings, such as methyl and methylene hydrogens or
            halogens, but atoms without siblings such as hydroxyl hydrogens, or phenyl
            hydrogens will remain in the backbone. Depth 2 includes bi-connected
            substituents such as phenyl rings (which contain internal symmetry) or
            multiple identical substituents attached to the same parent atom (which
            also contains symmetry), such as geminal diols (-C(OH)2) or other geminal
            substituents. Depth > 2 recursively peels more layers of substituents until
            the desired depth is reached or no further pendants exist.

    Returns:
        Dictionary mapping parent_idx (or None for backbone) to Component.
    """
    if depth == 0:
        idxs = np.array(range(len(s1.symbols)))  # Whole structure as backbone
        return {None: Component(parent_idx=None, pool=False, idxs=idxs)}

    # Depth == 1: identify singly attached pendants
    batches = [[], []]

    # Identify pendant groups
    pendant_groups: dict[int, list[int]] = defaultdict(list)  # (parent_idx) -> list of atom indices  # fmt: skip
    for atom_idx, nbrs in s1.adjacency_dict.items():
        if len(nbrs) == 1:  # Single attached pendant
            pendant_groups[nbrs[0]].append(atom_idx)

    backbone_indices = set(range(len(s1.symbols)))  # Initialize to all indices
    factored = {}

    # Build components: pendants first, then backbone. Sorting for deterministic output.
    for parent_idx, pendant_idxs in pendant_groups.items():
        if len(pendant_idxs) > 1:  # Multiple atoms in this pendant group
            # Sorting by symbol for element-wise cross-structure pool matching
            symbols = [s1.symbols[i] for i in pendant_idxs]
            sorted_pairs = sorted(zip(symbols, pendant_idxs))
            component = Component(
                parent_idx=parent_idx,
                pool=True,
                idxs=np.array([idx for _, idx in sorted_pairs]),
            )
            factored[parent_idx] = component
            backbone_indices.difference_update(pendant_idxs)

    # Add backbone component
    factored[None] = Component(
        parent_idx=None, pool=False, idxs=np.array(sorted(backbone_indices))
    )
    return factored


def _factored_to_batches(factored: dict[int | None, Component]) -> list[Batch]:
    """
    Convert the factored dictionary to batches.

    Returns:
        List of batches, where the first batch is the backbone and the subsequent batch
            contains the pendant groups.
    """
    batches = [[], []]
    for parent_idx, component in factored.items():
        if parent_idx is None:
            batches[0].append(component)
        else:
            batches[1].append(component)
    return batches


def _score_fixed_plus_component(
    A_geom: np.ndarray,
    B_geom: np.ndarray,
    P: np.ndarray,
    A_C: np.ndarray,
    A_fixed: np.ndarray,
    B_idx_perm: np.ndarray,
) -> float:
    """Compute RMSD after aligning A to B using fixed + component indices."""
    # score on fixed ∪ C, with a per-candidate temporary alignment
    A_idx_fixed = np.flatnonzero(A_fixed)
    A_fixed_plus_C = np.concatenate([A_idx_fixed, A_C])
    B_fixed_plus_C = np.concatenate([P[A_idx_fixed], B_idx_perm])

    # Align only the selected subset; faster than _align_known on full structure
    A_sel = A_geom[A_fixed_plus_C]
    B_sel = B_geom[B_fixed_plus_C]
    R, cA, cB = kabsch(A_sel, B_sel)
    return compute_rmsd((A_sel - cA) @ R.T + cB, B_sel, align=False)


def snap_rmsd(
    s1: Structure,
    s2: Structure,
    align: bool = True,
    factor: str | Callable[[Any], list[Batch]] = "pendant",
    factor_depth: int = 1,
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
        factor_depth: The depth of pendant groups to consider (default is 1).
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
    P = np.array(_map_via_canonical_labels(A, B))
    A_fixed = np.zeros(len(s1.symbols), dtype=bool)  # track fixed indices

    # --- Compute automorphism group of A
    A_gens, A_mantisa, A_exponent, A_coloring, A_num_orbits = pn.autgrp(A)

    A_geom = s1.geometry.copy()  # Mutable copy for alignment
    B_geom = s2.geometry  # Read-only

    # --- 2. Factor batches ----------------------------------------------------------
    if isinstance(factor, str):
        if factor.lower() == "pendant":
            A_factored = _pendant_factor(s1, depth=factor_depth)
            B_factored = _pendant_factor(s2, depth=factor_depth)
        else:
            raise ValueError(f"Unknown factoring method '{factor}'.")
    else:
        A_factored = factor(s1)  # expected: list[Batch]
    batches = _factored_to_batches(A_factored)  # ensure Batch format

    # --- 3. Main control loop -----------------------------------------------------
    for b_idx, batch in enumerate(batches):
        for c_idx, component in enumerate(batch):
            best_rmsd = float("inf")
            best_perm: list[int] | None = None
            A_C = component.idxs

            # TODO: Need to make sure component idxs are in canonical order, I think!
            # if component.parent_idx is None:  # Backbone component
            #     B_C = B_factored[None].idxs
            # else: # Pendant components
            #     B_C = B_factored[P[component.parent_idx]].idxs

            # Get corresponding B_C for this component
            if component.pool:  # Parent may have been remapped
                B_C = B_factored[P[component.parent_idx]].idxs
            # TODO: I think this is wrong. We'll need to do a lookup of the parent component
            # except for just backbone components, which can use the same lookup via None...
            # There's something to do with order here that matters. Order for the pool was
            # critical. For non-pool I'm not sure if we can somehow use the original canonical
            # order or something else... But I think we will need to look up the parent component
            # and find its indices in B to get the correct B_C.
            else:
                # Non-pool pendant or backbone: use P directly
                B_C = P[A_C]

            for perm in component_permutations(A_adj, A_coloring, A_fixed, component):
                B_idx_perm = B_C[perm]

                if align and (realign_per_component or (b_idx == 0 and c_idx == 0)):
                    # Align per-candidate; use fixed + component indices
                    score = _score_fixed_plus_component(A_geom, B_geom, P, A_C, A_fixed, B_idx_perm)  # fmt: skip
                else:
                    # No alignment; direct RMSD on component given established frame
                    score = compute_rmsd(A_geom[A_C], B_geom[B_idx_perm], align=False)

                if score < best_rmsd:
                    best_rmsd = score
                    best_perm = perm

            # Lock-in best permutation for this component
            P[A_C] = B_C[best_perm]
            A_fixed[A_C] = True

            # Establish common reference frame after first component in first batch
            if align and b_idx == 0 and c_idx == 0:
                A_geom = _align_known(A_geom, B_geom, P, A_fixed)

        # Batch-level global realignment; avoid duplicate alignments
        if align and not realign_per_component and not (b_idx == 0 and c_idx == 0):
            A_geom = _align_known(A_geom, B_geom, P, A_fixed)

    assert np.all(A_fixed), "Not all indices got fixed!."  # Sanity check
    count = Counter(P)
    assert all(v == 1 for v in count.values()), "P is not a valid permutation!"  # Sanity check while debugging # fmt: skip
    # Align false because already aligned at the end of the last batch if needed
    final_rmsd = compute_rmsd(A_geom, B_geom[P], align=False)
    return float(final_rmsd), P.tolist()


# def randomly_reorder_structure(struct: Structure) -> Structure:
#     # Create a random permutation of the indices
#     n = len(struct.symbols)
#     perm = np.random.permutation(n)  # New -> Old

#     # Build inverse: old_index -> new_index
#     inv = np.empty(n, dtype=int)
#     inv[perm] = np.arange(n)

#     new_symbols = []
#     new_geometry = []
#     new_connectivity = []
#     for new_idx, old_idx in enumerate(perm):
#         new_symbols.append(struct.symbols[old_idx])
#         new_geometry.append(struct.geometry[old_idx])

#     # for from_idx, to_idx, bo in struct.connectivity:
#     #     new_bond = (inv[from_idx], inv[to_idx], bo)
#     #     new_connectivity.append(new_bond)

#     # Reindex connectivity triplets; normalize endpoints so i <= j
#     if getattr(struct, "connectivity", None) is not None:
#         new_conn = []
#         for i_old, j_old, order in struct.connectivity:
#             i_new = int(inv[int(i_old)])
#             j_new = int(inv[int(j_old)])
#             if i_new > j_new:
#                 i_new, j_new = j_new, i_new
#             new_conn.append((i_new, j_new, float(order)))

#     # Create a new structure with the same other attributes, but updated ordering.
#     new_struct_dict = struct.model_dump()
#     new_struct_dict["symbols"] = new_symbols
#     new_struct_dict["geometry"] = new_geometry
#     new_struct_dict["connectivity"] = new_connectivity
#     return Structure.model_validate(new_struct_dict)


def randomly_reorder_structure(struct: "Structure") -> "Structure":
    n = len(struct.symbols)
    perm = np.random.permutation(n).astype(int)  # new_index -> old_index

    # inverse: old_index -> new_index
    inv = np.empty(n, dtype=int)
    inv[perm] = np.arange(n, dtype=int)

    d: dict[str, Any] = deepcopy(struct.model_dump())

    # Per-atom fields (add others your graph/coloring uses)
    d["symbols"] = [struct.symbols[old] for old in perm]
    d["geometry"] = np.asarray(struct.geometry)[perm]

    # Reindex connectivity triplets; normalize endpoints so i <= j
    if getattr(struct, "connectivity", None) is not None:
        new_conn = []
        for i_old, j_old, order in struct.connectivity:
            i_new = int(inv[int(i_old)])
            j_new = int(inv[int(j_old)])
            if i_new > j_new:
                i_new, j_new = j_new, i_new
            new_conn.append((i_new, j_new, float(order)))
        d["connectivity"] = new_conn

    return Structure.model_validate(d)


def _struct_to_rustworkx_graph(struct: "Structure") -> "rx.PyGraph":
    import rustworkx as rx

    g = rx.PyGraph()
    nodes = [g.add_node(sym) for sym in struct.symbols]
    for i, row in enumerate(struct.adjacency_matrix):
        for j in np.nonzero(row)[0]:
            if i < j:  # avoid adding duplicate edges
                g.add_edge(i, j, struct.adjacency_matrix[i, j])
    return g


if __name__ == "__main__":
    from time import time

    import rustworkx as rx
    from qcio import Structure
    from spyrmsd.rmsd import symmrmsd

    from qcinf import smiles_to_structure
    from qcinf.algorithms.geometry import rmsd

    def benchmark_snap(s1, s2, align=True, factor="pendant", **kwargs):
        start = time()
        s_rmsd, perm = snap_rmsd(s1, s2, align=align, factor=factor, **kwargs)
        snap_time = time() - start
        print(f"{s_rmsd:.5f}", perm)
        print(f"snapRMSD time: {snap_time:.4f} s")
        return s_rmsd, perm

    def benchmark_rdkit(s1, s2):
        start = time()
        rdkit_rmsd = rmsd(s1, s2, backend="rdkit")
        rdkit_time = time() - start
        print(f"{rdkit_rmsd:.5f}")
        print(f"RDKit RMSD time: {rdkit_time:.4f} s")
        return rdkit_rmsd

    def benchmark_symmrmsd(s1, s2, align=True):
        start = time()
        spyrmsd_rmsd = symmrmsd(
            s1.geometry,
            s2.geometry,
            s1.symbols,
            s2.symbols,
            s1.adjacency_matrix,
            s2.adjacency_matrix,
            center=True,
            minimize=align,
            cache=False,
        )
        spyrmsd_rmsd_time = time() - start
        print(f"{spyrmsd_rmsd:.5f}")
        print(f"spyrmsd RMSD time: {spyrmsd_rmsd_time:.4f} s")
        return spyrmsd_rmsd

    # smiles = "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C"
    smiles = "FC(F)(F)C1=CC(=CC(NC(=O)NC2=CC(=CC(=C2)C(F)(F)F)C(F)(F)F)=C1)C(F)(F)F"
    # smiles = "CCCCC"
    s1 = smiles_to_structure(smiles)
    s2 = smiles_to_structure(smiles)
    # s2 = smiles_to_structure(smiles)
    # s2 = smiles_to_structure(smiles)
    # s2 = smiles_to_structure(smiles)
    # s2 = smiles_to_structure(smiles)
    s2 = randomly_reorder_structure(s2)
    # s1.save("s1.json")
    # s2.save("s2.json")
    # s1 = Structure.open("s1.json")
    # s2 = Structure.open("s2.json")

    # g1 = rx.PyGraph.from_adjacency_matrix(s1.adjacency_matrix)
    # g1 = _struct_to_rustworkx_graph(s1)
    # g2 = _struct_to_rustworkx_graph(s2)
    # assert rx.is_isomorphic(g1, g2), "Rustworkx says not isomorphic!"

    # batches = _pendant_factor(s1)
    s_rmsd, perm = benchmark_snap(
        s1, s2, align=True, factor="pendant", realign_per_component=True
    )
    s_rmsd, perm = benchmark_snap(s1, s2, align=True, factor_depth=0)
    rdkit_rmsd = benchmark_rdkit(s1, s2)
    # spyrmsd_rmsd = benchmark_symmrmsd(s1, s2, align=True)
    # raw = compute_rmsd(s1.geometry, s2.geometry, align=False)
    # print(raw)
    # Compare s1_geom to s1.geometry to ensure no mutation
