# bc_tree_min.py
from __future__ import annotations

from typing import List

import numpy as np
import rustworkx as rx

from qcinf import smiles_to_structure


def graph_from_structure(structure) -> rx.PyGraph:
    """
    Build an undirected graph directly from the adjacency matrix.
    The node weights are the node indices (rustworkx default for from_adjacency_matrix).
    """
    mat = np.asarray(structure.adjacency_matrix, dtype=float)
    return rx.PyGraph.from_adjacency_matrix(mat)  # edges get float weights from matrix


def block_cut_tree(g: rx.PyGraph) -> rx.PyGraph:
    # 1) Edge→component-id mapping
    bcc = rx.biconnected_components(g)  # mapping-like object

    # 2) Gather node sets per component id
    comp_atoms = {}
    for (u, v), cid in bcc.items():  # mapping: (u,v) -> comp_id
        comp_atoms.setdefault(cid, set()).update((u, v))

    # 3) Articulation points
    arts = set(rx.articulation_points(g))

    # 4) Build BC tree
    bc = rx.PyGraph(multigraph=False)
    b_idx = {
        cid: bc.add_node(("B", frozenset(nodes))) for cid, nodes in comp_atoms.items()
    }
    a_idx = {a: bc.add_node(("A", a)) for a in arts}

    # 5) Connect A nodes to B nodes that contain them
    for cid, nodes in comp_atoms.items():
        for a in nodes & arts:
            bc.add_edge(a_idx[a], b_idx[cid], None)

    return bc


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


# --- tiny demo (requires your smiles_to_structure) ---
if __name__ == "__main__":
    # Example:
    struct = smiles_to_structure("CCC")
    g = graph_from_structure(struct)
    bc = block_cut_tree(g)
    print("BC-tree nodes:", [bc[n] for n in range(bc.num_nodes())])
    for u, v in bc.edge_list():
        print("BC-tree edge:", bc[u], "<->", bc[v])
