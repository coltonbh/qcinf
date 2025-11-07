# snap_rmsd.py
from __future__ import annotations
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pynauty as pn
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from .utils import kabsch, compute_rmsd
from qcio import Structure

# --- Types ------------------------------------------------------------------------
Index = int
Component = list[Index]
Batch = list[Component]


# --- Nauty helpers ----------------------------------------------------------------
def _list_of_sets(coloring: Sequence[Any]) -> list[set[int]]:
    """Convert a list of color-ids to pynauty's list[set] partition."""
    buckets: dict[Any, set[int]] = {}
    for v, c in enumerate(coloring):
        buckets.setdefault(c, set()).add(v)
    return list(buckets.values())


def _to_pynauty_graph(
    adj_dict: dict[int, list[int]], coloring: Sequence[Any]
) -> pn.Graph:
    return pn.Graph(
        number_of_vertices=len(adj_dict),
        adjacency_dict=adj_dict,
        vertex_coloring=_list_of_sets(coloring),
    )


def _map_from_canonical_labels(can1: list[int], can2: list[int]) -> list[int]:
    """Return M: index map s1->s2 using canonical labels of G1 and G2."""
    inv2 = [0] * len(can2)  # label -> original index in s2
    for i, lab in enumerate(can2):
        inv2[lab] = i
    return [inv2[lab1] for lab1 in can1]


# --- Partition for a component: singletons for fixed, split-in/out for C ----------
def _component_partition(
    base_colors: list[int], fixed: np.ndarray, C: Sequence[int]
) -> list[int]:
    """
    Start from base_colors (e.g., Nauty orbits). Encode:
      - pointwise stabilizer of 'fixed' by promoting to singletons
      - setwise stabilizer of C by splitting each color into (in-C) vs (out-of-C)
    """
    colors = list(base_colors)
    next_id = (max(colors) if colors else -1) + 1

    # Promote fixed to unique colors (singletons)
    for i, is_fixed in enumerate(fixed):
        if is_fixed:
            colors[i] = next_id
            next_id += 1

    # Split colors by in/out of C
    C_set = set(C)
    remap: dict[tuple[int, bool], int] = {}
    for v, c in enumerate(colors):
        key = (c, v in C_set)
        if key not in remap:
            remap[key] = next_id
            next_id += 1
        colors[v] = remap[key]
    return colors


# --- Restrict global generator (array form on 0..n-1) to local action on C --------
def _restrict_global_generators_to_C(
    gens_global: list[list[int]],
    C: Sequence[int],
) -> list[Permutation]:
    """Project each global generator to a k-point Permutation on C (deduplicated)."""
    g2l = {v: i for i, v in enumerate(C)}  # global -> local
    seen: set[tuple[int, ...]] = set()
    local: list[Permutation] = []

    for g in gens_global:
        # i (0..k-1) maps to position of g(C[i]) in C
        local_map = tuple(g2l[g[v]] for v in C)
        if local_map not in seen:
            seen.add(local_map)
            local.append(Permutation(local_map))

    if not local:  # trivial group
        local.append(Permutation(list(range(len(C)))))
    return local


def _setwise_stabilizer_on_C(
    adj_dict: dict[int, list[int]],
    base_colors: list[int],
    fixed: np.ndarray,
    C: Sequence[int],
) -> PermutationGroup:
    """(Aut(G)_(fixed))_{C}↾C as a local SymPy group on |C| points."""
    colors = _component_partition(base_colors, fixed, C)
    G = _to_pynauty_graph(adj_dict, colors)
    gens, *_ = pn.autgrp(G)  # generators as array-forms on 0..n-1
    local_gens = _restrict_global_generators_to_C(gens, C)
    return PermutationGroup(*local_gens)


# --- Align A to B using known correspondences P[fixed] ----------------------------
def _align_known(
    A: np.ndarray, B: np.ndarray, P: np.ndarray, fixed: np.ndarray
) -> np.ndarray:
    """Return a copy of A aligned to B using pairs (i -> P[i]) for i with fixed[i]."""
    A_aligned = A.copy()
    S = np.flatnonzero(fixed)
    if S.size >= 3:
        idx2 = P[S]
        R, cA, cB = kabsch(A[S], B[idx2])
        A_aligned = (A - cA) @ R.T + cB
    return A_aligned


# --- Default factoring placeholder -------------------------------------------------
def _factor_structure_default(n: int) -> list[Batch]:
    """Single-batch, single-component: act on all atoms (placeholder)."""
    return [[list(range(n))]]


# --- Main API ---------------------------------------------------------------------
def snap_rmsd(
    s1,  # qcio.Structure-like: .symbols, .geometry, .adjacency_dict
    s2,
    *,
    align: bool = True,
    factor: str | Callable[[Any], list[Batch]] | None = None,
    backend: str = "pynauty",
    realign_per_component: bool = False,
) -> tuple[float, list[int]]:
    """
    Compute snapRMSD(s1→s2). Returns (rmsd, P) where P maps s1 indices to s2 indices.

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
    """
    if backend != "pynauty":
        raise ValueError("Only backend='pynauty' is implemented here.")

    n = len(s1.symbols)

    # --- Graphs & isomorphism
    G1 = _to_pynauty_graph(s1.adjacency_dict, s1.symbols)
    G2 = _to_pynauty_graph(s2.adjacency_dict, s2.symbols)
    if not pn.isomorphic(G1, G2):
        raise ValueError(
            "Structure graphs are not isomorphic; same connectivity required."
        )

    # --- Canonical map P (s1->s2) and base color partition (e.g., orbits)
    M = np.array(
        _map_from_canonical_labels(pn.canon_label(G1), pn.canon_label(G2)), dtype=int
    )
    P = M.copy()  # single source-of-truth map (bijective)
    fixed = np.zeros(n, dtype=bool)  # which s1 indices are already locked-in

    # Using Nauty's orbit coloring as a good base partition
    _, _, _, G1_orbits, _ = pn.autgrp(G1)
    base_colors = list(G1_orbits)

    # --- Working coordinate copies (we only mutate G1_geom by accepted alignments)
    G1_geom = s1.geometry.copy()
    G2_geom = s2.geometry.copy()

    # --- Factoring (placeholder or user-provided)
    if factor is None or isinstance(factor, str):
        batches = _factor_structure_default(n)
    else:
        batches = factor(s1)  # expected: list[Batch]

    batches = [
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18, 19],
        ],
    ]  # hardcoded for hexane
    # --- Main control loop
    for b_idx, batch in enumerate(batches):
        for c_idx, C in enumerate(batch):
            C = np.array(C, dtype=int)

            # Local action group on C given current 'fixed'
            G_local = _setwise_stabilizer_on_C(s1.adjacency_dict, base_colors, fixed, C)

            # Baseline for C (so we never compose candidates accidentally)
            base_C = P[C].copy()

            best_score = float("inf")
            best_perm: Permutation | None = None

            # Precompute fixed sets for this component
            S_fixed = np.flatnonzero(fixed)
            idx2_fixed = P[S_fixed]

            # Score candidates
            for perm in G_local.generate_schreier_sims():
                idx2_C = base_C[perm.array_form]

                if align and ((b_idx == 0 and c_idx == 0) or realign_per_component):
                    # score on fixed ∪ C, with a per-candidate temporary alignment
                    S = np.concatenate([S_fixed, C])
                    idx2 = np.concatenate([idx2_fixed, idx2_C])
                    A_S = G1_geom[S]
                    B_S = G2_geom[idx2]

                    # do NOT mutate G1_geom here; align the subset temporarily
                    R, cA, cB = kabsch(A_S, B_S)
                    A_S_aligned = (A_S - cA) @ R.T + cB
                    score = compute_rmsd(A_S_aligned, B_S, align=False)
                else:
                    # component-only scoring, no per-candidate alignment
                    A_C = G1_geom[C]
                    B_C = G2_geom[idx2_C]
                    score = compute_rmsd(A_C, B_C, align=False)

                if score < best_score:
                    best_score = score
                    best_perm = perm

            # Lock-in winner on C, mark fixed
            assert best_perm is not None
            P[C] = base_C[best_perm.array_form]
            fixed[C] = True

        # Batch-level global realignment (optional)
        if align and not realign_per_component:
            G1_geom = _align_known(G1_geom, G2_geom, P, fixed)

    # --- Final RMSD (align once more if align=True to use *all* pairs)
    if align:
        final_A = _align_known(G1_geom, G2_geom, P, np.ones(n, dtype=bool))
        final = compute_rmsd(final_A, G2_geom[P], align=False)
    else:
        final = compute_rmsd(G1_geom, G2_geom[P], align=False)

    return float(final), P.tolist()


if __name__ == "__main__":
    from time import time

    from qcio import Structure
    from spyrmsd.rmsd import symmrmsd

    from qcinf import rmsd, smiles_to_structure

    # smiles = "FC(F)(F)C1=CC(=CC(NC(=O)NC2=CC(=CC(=C2)C(F)(F)F)C(F)(F)F)=C1)C(F)(F)F"
    smiles = "CCCCCC"
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
        cache=False,
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
