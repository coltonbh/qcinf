"""Block Cut Tree (BCT) algorithm for molecular fragmentation (hacking)."""

# bc_tree_substituents.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import rustworkx as rx

from qcinf import smiles_to_structure


@dataclass(frozen=True)
class BCNode:
    """A node in the Block–Cut tree."""

    kind: str  # 'B' for block, 'A' for articulation vertex
    payload: Any  # for 'B': frozenset of atom indices; for 'A': int atom index


@dataclass
class Substituent:
    anchor_atom: int  # articulation vertex that remains after collapse
    atoms_to_collapse: Set[int]  # atoms removed/abstracted into a pendant symbol
    blocks: List[
        frozenset
    ]  # the BCC node payloads that were collapsed (debug/inspection)


# -----------------------------
# 2) Block–Cut tree construction
# -----------------------------
def block_cut_tree(g: rx.PyGraph) -> Tuple[rx.PyGraph, Dict[int, int], Dict[int, int]]:
    """
    Returns:
      bc: PyGraph whose nodes are BCNode(kind='B', payload=frozenset(atom_ids))
                         or  BCNode(kind='A', payload=atom_id)
      a_index: mapping original atom_id -> BC-tree node index (for articulation vertices only)
      b_index: mapping block_id        -> BC-tree node index
    """
    # (a) Biconnected components as an edge->component-id mapping
    bcc_map = rx.biconnected_components(
        g
    )  # mapping of (u,v) -> comp_id  (order-insensitive)  # :contentReference[oaicite:1]{index=1}
    # Group edges by component id, then collect atoms per component
    import pdb

    pdb.set_trace()
    comp_edges: Dict[int, List[Tuple[int, int]]] = {}
    for u, v, _eid in g.edge_list():
        cid = bcc_map.get((u, v))
        if cid is None:
            cid = bcc_map.get((v, u))
        if cid is None:
            raise RuntimeError(
                "Edge not found in biconnected_components map; check rustworkx version."
            )
        comp_edges.setdefault(cid, []).append((u, v))

    comp_atoms: Dict[int, Set[int]] = {cid: set() for cid in comp_edges}
    for cid, edges in comp_edges.items():
        for u, v in edges:
            comp_atoms[cid].add(u)
            comp_atoms[cid].add(v)

    # (b) Articulation points
    arts: Set[int] = set(
        rx.articulation_points(g)
    )  # :contentReference[oaicite:2]{index=2}

    # (c) Build the BC tree
    bc = rx.PyGraph(multigraph=False)

    # Add 'B' nodes (blocks) with payload = frozenset(atom_ids)
    b_index: Dict[int, int] = {}
    for cid, nodes in comp_atoms.items():
        b_index[cid] = bc.add_node(BCNode(kind="B", payload=frozenset(nodes)))

    # Add 'A' nodes (articulation vertices)
    a_index: Dict[int, int] = {}
    for a in arts:
        a_index[a] = bc.add_node(BCNode(kind="A", payload=a))

    # Connect A-nodes to B-nodes when articulation atom is in that block
    for cid, nodes in comp_atoms.items():
        for a in nodes.intersection(arts):
            bc.add_edge(a_index[a], b_index[cid], None)

    return bc, a_index, b_index


# -----------------------------
# 3) Utilities on the BC tree
# -----------------------------
def _is_block_node(bc: rx.PyGraph, n: int) -> bool:
    return isinstance(bc[n], BCNode) and bc[n].kind == "B"


def _is_art_node(bc: rx.PyGraph, n: int) -> bool:
    return isinstance(bc[n], BCNode) and bc[n].kind == "A"


def degree(bc: rx.PyGraph, n: int) -> int:
    return len(bc.neighbors(n))


def block_nodes(bc: rx.PyGraph) -> List[int]:
    return [n for n in range(bc.num_nodes()) if _is_block_node(bc, n)]


def art_nodes(bc: rx.PyGraph) -> List[int]:
    return [n for n in range(bc.num_nodes()) if _is_art_node(bc, n)]


def block_is_leaf(bc: rx.PyGraph, b: int) -> bool:
    """Leaf blocks correspond to pendant biconnected components (phenyl, edge CH, etc.)."""
    return _is_block_node(bc, b) and degree(bc, b) == 1


# -----------------------------
# 4) Enumerate "substituent = leaf-subtree" collapses
# -----------------------------
def enumerate_substituents(bc: rx.PyGraph) -> List[Substituent]:
    """
    A unified pendant finder:
      - For rings (phenyl): a single leaf 'B' node attached to an articulation 'A'
      - For tree-like groups (methyl): several leaf 'B' nodes (each an edge) hang off one 'A'
      - For methylene in backbone: its 'A' won't be boundary of a leaf-subtree, so it won't be emitted.

    Returns a list of Substituent(anchor_atom, atoms_to_collapse, blocks)
    """
    # Precompute which block nodes are "internal" vs "leaf" in the BC tree
    B_nodes = block_nodes(bc)
    leaf_B = {b for b in B_nodes if block_is_leaf(bc, b)}

    # For each articulation node, see if it is the root of an *external* leaf-subtree:
    # An articulation A is an anchor if at least one of its incident B neighbors is a non-leaf "core" block
    # and the rest (if any) are leaf blocks we should collapse.
    subs: List[Substituent] = []

    for a in art_nodes(bc):
        nbrs = list(bc.neighbors(a))
        B_nbrs = [n for n in nbrs if _is_block_node(bc, n)]
        if not B_nbrs:
            continue

        # Partition A's B-neighbors into internal (deg>1) and leaf (deg==1)
        core_side = [b for b in B_nbrs if degree(bc, b) > 1]
        outer_leafs = [b for b in B_nbrs if b in leaf_B]

        # If there is exactly one core-side B, then everything else beyond A is a pendant subtree to collapse
        if len(core_side) == 1 and len(outer_leafs) >= 1:
            anchor_atom = bc[a].payload  # articulation atom id
            atoms_to_collapse: Set[int] = set()
            collapsed_blocks: List[frozenset] = []

            # Collect all B leaf blocks reachable from 'a' that are *not* the unique core-side neighbor.
            # This naturally groups: phenyl (single leaf B with many atoms) and methyl (several leaf B edges).
            for b in outer_leafs:
                payload = bc[b].payload  # frozenset(atom_ids) in that block
                collapsed_blocks.append(payload)
                atoms_to_collapse.update(payload)

            # Keep the anchor atom itself (do not collapse it)
            atoms_to_collapse.discard(anchor_atom)

            subs.append(
                Substituent(
                    anchor_atom=anchor_atom,
                    atoms_to_collapse=atoms_to_collapse,
                    blocks=collapsed_blocks,
                )
            )

    return subs


# -----------------------------
# 5) Convenience wrapper for Structure
# -----------------------------
def bc_tree_and_substituents_from_structure(structure):
    """
    Returns:
      bc: the BC tree as a rustworkx.PyGraph with BCNode payloads
      substituents: List[Substituent] (anchor atom id and atoms-to-collapse)
    """
    g = graph_from_structure(structure)
    bc, a_index, b_index = block_cut_tree(g)
    substituents = enumerate_substituents(bc)
    return bc, substituents


# -----------------------------
# Example usage (pseudo):
# -----------------------------
if __name__ == "__main__":
    struct = smiles_to_structure("CCC")
    bc, substituents = bc_tree_and_substituents_from_structure(struct)
    for s in substituents:
        print("Anchor:", s.anchor_atom, "Collapse:", sorted(s.atoms_to_collapse))
