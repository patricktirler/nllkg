from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

@dataclass
class InstanceGroup:
    """Grouped keypoints represented explicitly by arrays.
    node_ids: (K,) global indices of the keypoints in this instance.
    keypoint_scores: (K,) scores corresponding to node_ids order.
    adjacency_matrix: (K, K, R) relation scores subset for these nodes.
    keypoint_labels: Optional (K,) labels corresponding to node_ids order.
    keypoint_coords: Optional (K, 2) coordinates for each keypoint (e.g., pixel coords).
    keypoint_label_names: Optional (K,) string names corresponding to keypoint labels.
    """
    node_ids: np.ndarray
    keypoint_scores: np.ndarray
    adjacency_matrix: np.ndarray
    keypoint_labels: Optional[np.ndarray] = None
    keypoint_coords: Optional[np.ndarray] = None
    keypoint_label_names: Optional[np.ndarray] = None


# CheckMergeFn is a callable that takes two InstanceGroup objects, two node indices (u, v),
# and a relation type integer, and returns True iff the groups may be merged.
CheckMergeFn = Callable[[InstanceGroup, InstanceGroup, int, int, int], bool]


def group_keypoints_into_instances(
    keypoint_scores: np.ndarray,
    relation_scores: np.ndarray,
    check_merge: CheckMergeFn,
    keypoint_labels: Optional[np.ndarray] = None,
    keypoint_coords: Optional[np.ndarray] = None,
    keypoint_label_names: Optional[np.ndarray] = None,
    keypoint_score_threshold: float = 0.0,
    min_edge_score: float = 0.0,
) -> List[InstanceGroup]:
    """Group keypoints using a single descending pass over relation scores.

    Args:
        keypoint_scores: (N,) scores.
        relation_scores: (N, N, R) relation score tensor (R >= 1).
        check_merge: (grp_u, grp_v, u, v, rel_type) -> bool. If True, groups are merged.
                    u, v are global node indices. rel_type is argmax relation type for (u,v).
        keypoint_labels: Optional (N,) integer labels.
        keypoint_coords: Optional (N,2) coordinates corresponding to each keypoint.
        keypoint_label_names: Optional (N,) string names corresponding to keypoint labels.
        keypoint_score_threshold: Minimum score for keypoints to be considered.
        min_edge_score: Minimum edge score required to consider merging two groups.

    Returns:
        List of final InstanceGroup objects.
    """
    # TODO: This algorithm groups greedily based on max relation scores.
    # This means connectivity between two groups is determined by the maximum scoring edge only,
    # which may not be optimal. A more robust approach could consider the average of all edges between two groups.


    assert keypoint_labels is None or keypoint_labels.ndim == 1
    assert keypoint_scores.ndim == 1
    if keypoint_labels is not None:
        assert keypoint_scores.shape[0] == keypoint_labels.shape[0]
    assert relation_scores.ndim == 3, "relation_scores must be (N,N,R)"
    N, N2, R = relation_scores.shape
    assert N == N2, "relation_scores must be square in first two dims"
    if keypoint_coords is not None:
        assert keypoint_coords.shape == (N, 2), "keypoint_coords must be (N,2)"
    
    # Filter keypoints by score threshold
    keep = keypoint_scores >= keypoint_score_threshold
    keypoint_scores = keypoint_scores[keep]
    if keypoint_labels is not None:
        keypoint_labels = keypoint_labels[keep]
    if keypoint_coords is not None:
        keypoint_coords = keypoint_coords[keep]
    if keypoint_label_names is not None:
        keypoint_label_names = [keypoint_label_names[i] for i in np.nonzero(keep)[0]]
    relation_scores = relation_scores[np.ix_(keep, keep, np.arange(R))]
    N = keypoint_scores.shape[0]

    if N == 0:
        return []

    # Initialize singleton groups
    groups: List[InstanceGroup] = []
    for i in range(N):
        groups.append(InstanceGroup(
            node_ids=np.array([i], dtype=np.int32),
            keypoint_labels=(None if keypoint_labels is None else np.array([keypoint_labels[i]], dtype=keypoint_labels.dtype)),
            keypoint_scores=np.array([keypoint_scores[i]], dtype=keypoint_scores.dtype),
            adjacency_matrix=np.zeros((1, 1, R), dtype=relation_scores.dtype),
            keypoint_coords=(None if keypoint_coords is None else np.array([keypoint_coords[i]], dtype=keypoint_coords.dtype)),
            keypoint_label_names = (None if keypoint_label_names is None else np.array([keypoint_label_names[i]]))
        ))
    node_to_group = np.arange(N)  # global node id -> index into groups list

    # Build edge list with max score and relation type per pair (i<j)
    upper_i, upper_j = np.triu_indices(N, k=1)
    edge_relations = relation_scores[upper_i, upper_j, :]  # (E, R)
    best_relation_type = np.argmax(edge_relations, axis=1)  # (E,)
    best_relation_score = edge_relations[np.arange(edge_relations.shape[0]), best_relation_type]  # (E,)

    # Filter edges by min_edge_score
    valid_mask = best_relation_score >= min_edge_score
    sorted_indices = np.argsort(-best_relation_score[valid_mask])  # descending order

    sorted_i = upper_i[valid_mask][sorted_indices]
    sorted_j = upper_j[valid_mask][sorted_indices]
    sorted_relation_type = best_relation_type[valid_mask][sorted_indices]

    for u, v, rel_type in zip(sorted_i, sorted_j, sorted_relation_type):
        group_u_idx = node_to_group[u]
        group_v_idx = node_to_group[v]

        # if already in the same group, skip
        if group_u_idx == group_v_idx:
            continue

        group_u = groups[group_u_idx]
        group_v = groups[group_v_idx]
        if not check_merge(group_u, group_v, int(u), int(v), int(rel_type)):
            continue
        merged_group = _merge_groups(group_u, group_v, relation_scores)

        # Place merged group in slot group_u_idx, remove group_v_idx via swap-delete
        groups[group_u_idx] = merged_group
        for nid in merged_group.node_ids:
            node_to_group[int(nid)] = group_u_idx
        last_idx = len(groups) - 1
        if group_v_idx != last_idx:
            groups[group_v_idx] = groups[last_idx]
            for nid in groups[group_v_idx].node_ids:
                node_to_group[int(nid)] = group_v_idx
        groups.pop()

    return groups


def make_check_merge_max_label(max_per_label) -> CheckMergeFn:
    """
    Factory function to create a check_merge function that allows merging two groups only if the resulting group
    does not exceed the maximum allowed number of keypoints per label.

    Args:
        max_per_label (dict or int): If dict, maps label to max count. If int, all labels share the same max.

    Returns:
        CheckMergeFn: A function that returns True if merging the two groups would not violate per-label limits.
    """
    def check_merge(grp_u, grp_v, u, v, rel_type):
        merged_labels = np.concatenate([grp_u.keypoint_labels, grp_v.keypoint_labels])
        # Count occurrences per label
        unique, counts = np.unique(merged_labels, return_counts=True)
        if isinstance(max_per_label, int):
            if np.any(counts > max_per_label):
                return False
        else:
            for lbl, cnt in zip(unique, counts):
                if cnt > max_per_label.get(int(lbl), max(counts)):
                    return False
        return True
    return check_merge



def _merge_groups(
    grp_a: InstanceGroup,
    grp_b: InstanceGroup,
    relation_scores: Optional[np.ndarray] = None
) -> InstanceGroup:
    """
    Merge two InstanceGroup objects.

    Behavior:
    - node_ids, labels, and scores are concatenated preserving order: (grp_a nodes, grp_b nodes).
    - Intra-group adjacency (the two diagonal blocks) is copied exactly from grp_a / grp_b
      so any previously computed / refined values are preserved.
    - If relation_scores (global (N,N,R)) is provided, the cross-group relations
      (off-diagonal blocks) are filled from it:
          adj[:Ka, Ka:, :] = relation_scores[grp_a.node_ids][:, grp_b.node_ids, :]
          adj[Ka:, :Ka, :] = transpose of the above
      (intra-group blocks are NOT overwritten by global scores).
    - If relation_scores is None, cross blocks remain zero.

    Args:
        grp_a: First group.
        grp_b: Second group.
        relation_scores: Optional global relation score tensor (N,N,R).

    Returns:
        New merged InstanceGroup with combined metadata and adjacency.
    """
    new_node_ids = np.concatenate([grp_a.node_ids, grp_b.node_ids])
    new_labels = None if (grp_a.keypoint_labels is None or grp_b.keypoint_labels is None) else np.concatenate([grp_a.keypoint_labels, grp_b.keypoint_labels])
    new_scores = np.concatenate([grp_a.keypoint_scores, grp_b.keypoint_scores])

    Ka = grp_a.node_ids.size
    Kb = grp_b.node_ids.size
    R = grp_a.adjacency_matrix.shape[2]
    assert grp_b.adjacency_matrix.shape[2] == R, "Adjacency channel mismatch"

    # Initialize adjacency with zeros then place preserved intra-group blocks.
    adj = np.zeros((Ka + Kb, Ka + Kb, R), dtype=grp_a.adjacency_matrix.dtype)
    adj[:Ka, :Ka, :] = grp_a.adjacency_matrix
    adj[Ka:, Ka:, :] = grp_b.adjacency_matrix

    if relation_scores is not None:
        # Fill only cross-group relations from global tensor.
        # Shape: (Ka, Kb, R)
        cross = relation_scores[np.ix_(grp_a.node_ids, grp_b.node_ids, np.arange(R))]
        adj[:Ka, Ka:, :] = cross
        adj[Ka:, :Ka, :] = np.transpose(cross, (1, 0, 2))

    # Merge coords: if neither group has coords -> None, otherwise concatenate,
    # filling missing coords with NaNs to preserve ordering.
    if grp_a.keypoint_coords is None and grp_b.keypoint_coords is None:
        new_coords = None
    else:
        a_coords = grp_a.keypoint_coords if grp_a.keypoint_coords is not None else np.full((Ka, 2), np.nan, dtype=float)
        b_coords = grp_b.keypoint_coords if grp_b.keypoint_coords is not None else np.full((Kb, 2), np.nan, dtype=float)
        new_coords = np.concatenate([a_coords, b_coords], axis=0)

    # Merge label names: concatenate if present in either group.
    if grp_a.keypoint_label_names is None and grp_b.keypoint_label_names is None:
        new_label_names = None
    else:
        a_names = grp_a.keypoint_label_names if grp_a.keypoint_label_names is not None else np.full(Ka, "", dtype=object)
        b_names = grp_b.keypoint_label_names if grp_b.keypoint_label_names is not None else np.full(Kb, "", dtype=object)
        new_label_names = np.concatenate([a_names, b_names], axis=0)

    return InstanceGroup(
        node_ids=new_node_ids,
        keypoint_labels=new_labels,
        keypoint_scores=new_scores,
        adjacency_matrix=adj,
        keypoint_coords=new_coords,
        keypoint_label_names=new_label_names
    )



__all__ = [
    "InstanceGroup",
    "group_keypoints_into_instances",
    "CheckMergeFn",
    "make_check_merge_max_label",
]
