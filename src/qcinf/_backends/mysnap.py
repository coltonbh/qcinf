import itertools as it
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pynauty as pn
from qcio import Structure
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation

from .utils import compute_rmsd, kabsch, qcp_rotation, kabsch_numba
from .utils2 import qcp_rmsd_numba


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


def _to_pynauty_graph(adj_dict: dict[int, list[int]], coloring: list[Any]) -> pn.Graph:
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
    coloring: list[int], P_opt: np.ndarray, C: np.ndarray
) -> list[int]:
    """
    Start from base_colors (e.g., Nauty coloring). Encode:
      - pointwise stabilizer of 'fixed' by promoting to singletons
      - setwise stabilizer of C by splitting each color into (in-C) vs (out-of-C)
    """
    new_coloring = list(coloring)
    next_id = max(new_coloring) + 1

    # Promote fixed to unique colors (singletons)
    for i, is_fixed in enumerate(P_opt != -1):
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
    invA = [0] * len(canA)  # inv(can1): orig1 -> canon
    for i_canon, i_orig1 in enumerate(canA):
        invA[i_orig1] = i_canon
    # P[i_orig1] = can2[canon_of_i_orig1]
    return [canB[invA[i_orig1]] for i_orig1 in range(len(canA))]


def _restrict_global_generators_to_C(
    gens_global: list[list[int]],
    C: np.ndarray,
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
    P_opt: np.ndarray,
    C: np.ndarray,
) -> PermutationGroup:
    """(Aut(G)_(fixed))_{C}↾C as a local SymPy group on |C| points.

    This computes the setwise stabilizer of C within the pointwise stabilizer of
    singletons restricted to the action on C.

    Parameters:
        adj_dict: The adjacency dictionary of the structure graph.
        coloring: The vertex coloring (partition) for the whole structure.
        P_opt: Partial mapping of structure indices; -1 means unfixed.
        component: The list of indices in the current component.
    """
    # Fix already assigned indices and stabilize component setwise
    colors = _component_partition(base_colors, P_opt, C)
    G_c = _to_pynauty_graph(adj_dict, colors)
    gens, *_ = pn.autgrp(G_c)
    # Restrict to C
    local_gens = _restrict_global_generators_to_C(gens, C)
    return PermutationGroup(*local_gens)


def _align_fixed(A: np.ndarray, B: np.ndarray, P_opt: np.ndarray) -> np.ndarray:
    """Return a copy of A aligned to B using pairs (i -> P[i]) for i with fixed[i]."""
    fixed = P_opt != -1
    R, cA, cB = kabsch(A[fixed], B[P_opt[fixed]])
    R_q, cA_q, cB_q = qcp_rotation(A[fixed], B[P_opt[fixed]])
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
    P_opt: np.ndarray,
    component: Component,
) -> Iterable[list[int]]:
    """Aut(G)_{fixed}↾C as a local SymPy group on |C| points.

    This computes the pointwise stabilizer of the fixed vertices restricted to
    the action on C.

    Parameters:
        adj_dict: The adjacency dictionary of the structure graph.
        coloring: The vertex coloring (partition) for the whole structure.
        P_opt: Partial mapping of structure indices; -1 means unfixed.
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
            adj_dict, base_colors, P_opt, component.idxs
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
    # Identify pendant groups
    pendant_groups: dict[int, list[int]] = defaultdict(list)  # (parent_idx) -> list of atom indices  # fmt: skip
    for atom_idx, nbrs in s1.adjacency_dict.items():
        if len(nbrs) == 1:  # Single attached pendant
            pendant_groups[nbrs[0]].append(atom_idx)

    backbone_indices = set(range(len(s1.symbols)))  # Initialize to all indices
    factored: dict[int | None, Component] = {}

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
    batches: list[list[Component]] = [[], []]
    for parent_idx, component in factored.items():
        if parent_idx is None:
            batches[0].append(component)
        else:
            batches[1].append(component)
    return batches


def _align_and_score_fixed_plus_component(
    A_geom: np.ndarray,
    B_geom: np.ndarray,
    P_opt: np.ndarray,
    A_C: np.ndarray,
    B_idx_perm: np.ndarray,
) -> float:
    """Compute RMSD after aligning A to B using fixed + component indices."""
    # score on fixed ∪ C, with a per-candidate temporary alignment
    A_idx_fixed = np.flatnonzero(P_opt != -1)
    A_fixed_plus_C = np.concatenate([A_idx_fixed, A_C])
    B_fixed_plus_C = np.concatenate([P_opt[A_idx_fixed], B_idx_perm])

    # Align only the selected subset; faster than _align_known on full structure
    return compute_rmsd(A_geom[A_fixed_plus_C], B_geom[B_fixed_plus_C], align=True)
    # return qcp_rmsd_numba(A_geom[A_fixed_plus_C], B_geom[B_fixed_plus_C])


# def validate_factoring(
#     s: Structure,
#     batches_raw: list[list[list[int]]] | None,
# ) -> list[Batch]:
#     """
#     Validate a user-supplied factoring and convert it into an execution plan.

#     Parameters
#     ----------
#     s
#         Structure A whose indices the factoring refers to.
#     batches_raw
#         List of batches; each batch is a list of components; each component is a
#         list of atom indices. batches_raw[0] is assumed to be the backbone batch.

#     Returns
#     -------
#     batches : list[Batch]
#         Execution-plan batches where each Batch is a list[Component].
#         - Batch 0: exactly one backbone Component with parent_idx=None.
#         - Batches 1..: Components merged by parent attachment index.

#     Raises
#     ------
#     ValueError
#         If indices are out of range, duplicated, missing, or components
#         (except the backbone) do not have exactly one out-of-component attachment.
#     """
#     n_atoms = len(s.symbols)
#     adj = s.adjacency_dict

#     if not batches_raw:
#         # Default: whole structure is a single backbone component
#         backbone = Component(
#             parent_idx=None,
#             pool=False,
#             idxs=np.arange(n_atoms, dtype=int),
#         )
#         return [[[backbone]]]

#     # --- Global index validation ----------------------------------------------------
#     seen: set[int] = set()
#     for b_idx, batch in enumerate(batches_raw):
#         for c_idx, comp in enumerate(batch):
#             for idx in comp:
#                 if idx < 0 or idx >= n_atoms:
#                     raise ValueError(
#                         f"Index {idx} in batch {b_idx}, component {c_idx} "
#                         f"out of range for structure with {n_atoms} atoms."
#                     )
#                 if idx in seen:
#                     raise ValueError(
#                         f"Index {idx} appears in multiple components "
#                         f"(batch {b_idx}, component {c_idx})."
#                     )
#                 seen.add(idx)

#     if seen != set(range(n_atoms)):
#         missing = set(range(n_atoms)) - seen
#         raise ValueError(
#             f"Factoring does not cover the structure cleanly: missing indices {sorted(missing)}"
#         )

#     # --- Backbone (component 0) --------------------------------------------------
#     proposed_backbone = batches_raw[0][0]
#     if len(proposed_backbone) < 3:
#         raise ValueError(
#             "Backbone component must have at least 3 atoms to define a reference frame."
#         )
#     backbone_idxs = np.array(sorted(batches_raw[0][0]), dtype=int)
#     backbone_component = Component(
#         parent_idx=None,
#         pool=False,
#         idxs=backbone_idxs,
#     )

#     batches: list[Batch] = [[backbone_component]]

#     # --- Pendant batches (components 1+...) ---------------------------------------------
#     for b_idx, batch in enumerate(batches_raw):
#         # Group components by parent index
#         parent_to_indices: dict[int, list[int]] = defaultdict(list)

#         for c_idx, comp in enumerate(batch):
#             if b_idx == 0 and c_idx == 0:
#                 continue  # skip backbone component already processed

#             comp_set = set(comp)
#             parents: set[int] = set()

#             # Find all out-of-component neighbors
#             for idx in comp:
#                 for nbr in adj[idx]:
#                     if nbr not in comp_set:
#                         parents.add(nbr)

#             if not parents:
#                 raise ValueError(f"Component (batch {b_idx}, component {c_idx}) has no out-of-component attachment; expected exactly one parent.")  # fmt: skip
#             if len(parents) > 1:
#                 raise ValueError(f"Component (batch {b_idx}, component {c_idx}) has multiple parents {sorted(parents)}; factoring must be singly anchored.")  # fmt: skip

#             parent_idx = parents.pop()
#             parent_to_indices[parent_idx].extend(comp)

#         # Merge components that share the same parent
#         batch_components: Batch = []
#         for parent_idx, idxs in parent_to_indices.items():
#             uniq_sorted = np.array(sorted(set(idxs)), dtype=int)
#             batch_components.append(
#                 Component(
#                     parent_idx=parent_idx,
#                     pool=False,  # we'll detect pool vs structured at runtime
#                     idxs=uniq_sorted,
#                 )
#             )

#         batches.append(batch_components)

#     return batches

# def validate_factoring(
#     s: Structure,
#     batches_raw: list[list[list[int]]] | None,
# ) -> list[Batch]:
#     """
#     Validate a user-supplied factoring and convert it into an execution plan.

#     Parameters
#     ----------
#     s
#         Structure A whose indices the factoring refers to.
#     batches_raw
#         List of batches; each batch is a list of components; each component is a
#         list of atom indices. batches_raw[0] is assumed to be the backbone batch.

#     Returns
#     -------
#     batches : list[Batch]
#         Execution-plan batches where each Batch is a list[Component].
#         - Batch 0: exactly one backbone Component with parent_idx=None.
#         - Batches 1..: Components merged by parent attachment index.
#     Raises
#     ------
#     ValueError
#         If indices are out of range, duplicated, missing, or components
#         (except the backbone) do not have exactly one out-of-component attachment.
#     """
#     if batches_raw is None:
#         # Default: whole structure is a single backbone component
#         idxs = np.array(range(len(s.symbols)))  # Whole structure as backbone
#         backbone = Component(parent_idx=None, pool=False, idxs=idxs)
#         return [[[backbone]]]

#     for b_idx, batch in enumerate(batches_raw):
#         for c_idx, comp in enumerate(batch):
#             for idx in comp:
#                 if idx < 0 or idx >= len(s.symbols):
#                     raise ValueError(f"Index {idx} in batch {b_idx}, component {c_idx} out of range for structure with {len(s.symbols)} atoms.")  # fmt: skip


def snap_rmsd(
    s1: Structure,
    s2: Structure,
    align: bool = True,
    factor: str | dict[int | None, Component] = "pendant",
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
    # --- 1. Build graphs, canonical label map, compute automorphism group -----------
    # NOTE: Currently running nauty 3x on A and 2x on B. Can optimize?
    #   - Once per graph for isomorphism check
    #   - Once per graph for canonical labels
    #   - Once for A to get automorphism group

    # --- Graphs & isomorphism check
    A_adj = s1.adjacency_dict  # compute once since used multiple times
    if not A_adj or not s2.adjacency_dict:
        raise ValueError("Structures lack connectivity information.")

    A = _to_pynauty_graph(A_adj, s1.symbols)
    B = _to_pynauty_graph(s2.adjacency_dict, s2.symbols)

    if not pn.isomorphic(A, B):
        raise ValueError("Structures not isomorphic. Same connectivity required.")

    # --- Canonical labels & initial map
    M = np.array(_map_via_canonical_labels(A, B))
    P_opt = np.full(len(s1.symbols), -1, dtype=int)  # placeholder for final map

    # --- Compute automorphism group of A
    A_gens, A_mantissa, A_exponent, A_coloring, A_num_orbits = pn.autgrp(A)
    A_geom = s1.geometry  # Will not mutate s1.geometry
    B_geom = s2.geometry  # Will not mutate s2.geometry

    # --- 2. Factor batches ----------------------------------------------------------
    if isinstance(factor, str):
        if factor.lower() == "pendant":
            A_factored = _pendant_factor(s1, depth=factor_depth)
            B_factored = _pendant_factor(s2, depth=factor_depth)
        else:
            raise ValueError(f"Unknown factoring method '{factor}'.")
    else:
        A_factored = factor
    batches = _factored_to_batches(A_factored)  # ensure Batch format

    print(batches)

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
            p_idx = component.parent_idx
            if p_idx is None or P_opt[p_idx] == M[p_idx]:  # Backbone component or parent not remapped  # fmt: skip
                B_C = M[A_C]  # Initial canonical map still valid
            elif component.pool:  # Remapped pool pendant
                # Currently implemented in factored ordering; need to change to dynamic mapping here
                B_C = B_factored[P_opt[component.parent_idx]].idxs
            else:  # Remapped structured pendant
                # Implement with nauty canonical maps on local subgraphs rooted at parent_idx
                raise NotImplementedError(
                    "Remapped structured pendants not yet implemented."
                )

            for perm in component_permutations(A_adj, A_coloring, P_opt, component):
                B_idx_perm = B_C[perm]

                if align and (realign_per_component or (b_idx == 0 and c_idx == 0)):
                    # Align per-candidate; use fixed + component indices
                    score = _align_and_score_fixed_plus_component(A_geom, B_geom, P_opt, A_C,B_idx_perm)  # fmt: skip
                else:
                    # No alignment; direct RMSD on component given established frame
                    score = compute_rmsd(A_geom[A_C], B_geom[B_idx_perm], align=False)

                if score < best_rmsd:
                    best_rmsd = score
                    best_perm = perm

            # Lock-in best permutation for this component
            P_opt[A_C] = B_C[best_perm]

            # Per component and/or first component alignment (establishes frame)
            if align and (realign_per_component or (b_idx == 0 and c_idx == 0)):
                A_geom = _align_fixed(A_geom, B_geom, P_opt)

        # Realign after each batch; avoid duplicate alignments
        if align and not realign_per_component and not (b_idx == 0 and c_idx == 0):
            A_geom = _align_fixed(A_geom, B_geom, P_opt)

    assert np.all(P_opt != -1), (
        "Not all indices got fixed!."
    )  # Sanity check for debugging
    count = Counter(P_opt)
    assert all(v == 1 for v in count.values()), "P_opt is not a valid permutation!"  # Sanity check while debugging # fmt: skip
    # Align false because already aligned at the end of the last batch if needed
    final_rmsd = compute_rmsd(A_geom, B_geom[P_opt], align=False)
    return float(final_rmsd), P_opt.tolist()


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
    from time import time, perf_counter

    import rustworkx as rx
    from qcio import Structure
    from spyrmsd.rmsd import symmrmsd

    from qcinf import smiles_to_structure
    from qcinf.algorithms.geometry import rmsd
    from rmsd.calculate_rmsd import (
        centroid,
        kabsch_rmsd,
        reorder_inertia_hungarian,
        check_reflections,
    )
    from qcconst import periodic_table as pt
    from qcconst import constants as const

    def benchmark_rmsd(s1, s2, align=True):
        # center on centroids as in main()
        start = time()
        p_coord_sub = s1.geometry_angstrom - centroid(s1.geometry_angstrom)
        q_coord_sub = s2.geometry_angstrom - centroid(s2.geometry_angstrom)

        p_atomic_numbers = np.array([getattr(pt, sym).number for sym in s1.symbols])
        q_atomic_numbers = np.array([getattr(pt, sym).number for sym in s2.symbols])

        # Compute RMSD with reflections allowed
        rmsd_val, q_swap, q_reflection, q_review = check_reflections(
            p_atomic_numbers,
            q_atomic_numbers,
            p_coord_sub,
            q_coord_sub,
            reorder_method=reorder_inertia_hungarian,  # inertia + Hungarian
            rmsd_method=kabsch_rmsd,  # Kabsch rotation
            keep_stereo=False,  # like --use-reflections
        )
        rmsd_time = time() - start
        rmsd_val *= const.ANGSTROM_TO_BOHR
        print(f"{rmsd_val:.5f}")
        print(f"python RMSD time: {rmsd_time:.4f} s")
        return rmsd_val

    def benchmark_snap(s1, s2, align=True, factor="pendant", **kwargs):
        start = time()
        s_rmsd, perm = snap_rmsd(s1, s2, align=align, factor=factor, **kwargs)
        snap_time = time() - start
        print(f"{s_rmsd:.5f}", perm)
        print(f"snapRMSD time: {snap_time:.4f} s")
        return s_rmsd, perm, snap_time

    def benchmark_rdkit(s1, s2):
        start = time()
        rdkit_rmsd = rmsd(s1, s2, backend="rdkit")
        rdkit_time = time() - start
        print(f"{rdkit_rmsd:.5f}")
        print(f"RDKit RMSD time: {rdkit_time:.4f} s")
        return rdkit_rmsd, rdkit_time

    def benchmark_symmrmsd(s1, s2, align=True):
        # from spyrmsd.graph import _set_backend
        # _set_backend("networkx")
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

    def bench_pair(s1, s2, n=10000):
        A = s1.geometry
        B = s2.geometry

        # Warmup
        for _ in range(2):
            kabsch(A, B)
            qcp_rotation(A, B)
            kabsch_numba(A, B)
            compute_rmsd(A, B)
            qcp_rmsd_numba(A, B)

        # Kabsch
        t0 = perf_counter()
        for _ in range(n):
            kabsch(A, B)
        t_kabsch = perf_counter() - t0

        # QCP
        t0 = perf_counter()
        for _ in range(n):
            qcp_rotation(A, B)
        t_qcp = perf_counter() - t0

        # Kabsch Numba
        t0 = perf_counter()
        for _ in range(n):
            kabsch_numba(A, B)
        t_kabsch_numba = perf_counter() - t0

        # Regular RMSD
        t0 = perf_counter()
        for _ in range(n):
            rmsd_val = compute_rmsd(A, B)
        t_rmsd = perf_counter() - t0

        # RMSD Numba
        t0 = perf_counter()
        for _ in range(n):
            qcp_rmsd_val = qcp_rmsd_numba(A, B)

        t_rmsd_numba = perf_counter() - t0

        print(f"N = {A.shape[0]}")
        print(f"  kabsch:     {t_kabsch:.6f} s total, {t_kabsch / n:.3e} s/call")
        print(f"  qcp_rot:    {t_qcp:.6f} s total, {t_qcp / n:.3e} s/call")
        print(f"  kabsch_numba: {t_kabsch_numba:.6f} s total, {t_kabsch_numba / n:.3e} s/call")  # fmt: skip
        print(f"  rmsd:       {t_rmsd:.6f} s total, {t_rmsd / n:.3e} s/call")
        print(f"  rmsd_numba: {t_rmsd_numba:.6f} s total, {t_rmsd_numba / n:.3e} s/call")  # fmt: skip
        print(f"  rmsd_val: {rmsd_val:.6f}, qcp_rmsd_val: {qcp_rmsd_val:.6f}")

    # smiles = "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C"
    # smiles = "FC(F)(F)C1=CC(=CC(NC(=O)NC2=CC(=CC(=C2)C(F)(F)F)C(F)(F)F)=C1)C(F)(F)F"
    smiles = "CCCCC"
    s1 = smiles_to_structure(smiles)
    s2 = smiles_to_structure(smiles)
    s1.save("s1-routine.xyz")
    s1.save("s1-routine.json")
    s2.save("s2-routine.xyz")
    s2.save("s2-routine.json")
    # s1 = Structure.open("/home/cbh/dev/personal/qcinf/s1-pentane2.json")
    # s2 = Structure.open("/home/cbh/dev/personal/qcinf/s2-pentane2.json")

    # R, cA, cB = kabsch(s1.geometry, s2.geometry)
    # R_q, cA_q, cB_q = qcp_rotation(s1.geometry, s2.geometry)
    # bench_pair(s1, s2, n=1)

    # s2 = randomly_reorder_structure(s2)
    # s1.save("s1.json")
    # s2.save("s2.json")
    # s1 = Structure.open("s1.json")
    # s2 = Structure.open("s2.json")

    # g1 = rx.PyGraph.from_adjacency_matrix(s1.adjacency_matrix)
    # g1 = _struct_to_rustworkx_graph(s1)
    # g2 = _struct_to_rustworkx_graph(s2)
    # assert rx.is_isomorphic(g1, g2), "Rustworkx says not isomorphic!"

    # batches = _pendant_factor(s1)
    # python_rmsd = benchmark_rmsd(s1, s2, align=True)
    s_rmsd, perm, snap_time = benchmark_snap(s1, s2, align=True, factor="pendant")
    # s_rmsd, perm, snap_time_unfact = benchmark_snap(s1, s2, align=True, factor_depth=0)
    rdkit_rmsd, rdkit_time = benchmark_rdkit(s1, s2)
    spyrmsd_rmsd = benchmark_symmrmsd(s1, s2, align=True)
    # raw = compute_rmsd(s1.geometry, s2.geometry, align=False)
    # print(raw)
    # Compare s1_geom to s1.geometry to ensure no mutation
    # print(f"Speedup over RDKit: {rdkit_time / snap_time:.2f}x")

    import cProfile, pstats, io

    pr = cProfile.Profile()
    pr.enable()

    # the workload
    s_rmsd, perm = snap_rmsd(s1, s2, align=True, factor="pendant", factor_depth=1)

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")  # or "tottime"
    ps.print_stats(30)  # top 30
    # print(s.getvalue())
    pr.dump_stats("profile.out")
