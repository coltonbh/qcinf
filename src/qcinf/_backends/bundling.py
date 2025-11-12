"""ChatGPT Offer:
If you want, I can adapt your factoring to emit (component, anchor_id) so 
_bundle_components becomes a 10-liner and never touches the adjacency dict again.
"""

def _component_anchor(adj: dict[int, list[int]], comp: list[int]) -> int | None:
    C = set(comp)
    ext = set()
    for u in comp:
        for v in adj[u]:
            if v not in C:
                ext.add(v)
                if len(ext) > 1:
                    return None  # multi-anchored → not a pendant star
    return next(iter(ext)) if ext else None  # None if no external neighbor

def _is_pendant(adj: dict[int, list[int]], comp: list[int]) -> bool:
    C = set(comp)
    # all vertices have <=1 neighbor outside C
    for u in comp:
        count_out = sum(1 for v in adj[u] if v not in C)
        if count_out > 1:
            return False
    return True

def _bundle_components(
    batches: list[list[list[int]]],
    adj: dict[int, list[int]],
) -> list[list[list[int]]]:
    bundled: list[list[list[int]]] = []
    for batch in batches:
        # group by anchor id for those that are single-anchored pendants
        anchor_to_idxs: dict[int, list[int]] = {}
        for i, comp in enumerate(batch):
            if _is_pendant(adj, comp):
                a = _component_anchor(adj, comp)
                if a is not None:
                    anchor_to_idxs.setdefault(a, []).append(i)

        used = set()
        new_batch: list[list[int]] = []
        for i, comp in enumerate(batch):
            if i in used:
                continue
            # if this comp participates in a sibling group, bundle them
            a = _component_anchor(adj, comp) if _is_pendant(adj, comp) else None
            if a is not None and len(anchor_to_idxs.get(a, [])) >= 2:
                # union all siblings sharing anchor a
                sib_idxs = [j for j in anchor_to_idxs[a] if j not in used]
                union = sorted({v for j in sib_idxs for v in batch[j]})
                new_batch.append(union)
                used.update(sib_idxs)
            else:
                new_batch.append(sorted(comp))
                used.add(i)
        bundled.append(new_batch)
    return bundled
