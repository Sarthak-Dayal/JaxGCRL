"""A continuous point mass in a maze: positions in R^2, actions in [-1, 1]^2."""

import heapq

import numpy as np

from gridworld.world import maze_walls

GAMMA = 0.95


def build_world(n=11, seed=0, step=0.35, goal_radius=0.6, sub=3):
    """A point-mass maze.

    The walls come from the same recursive backtracker the discrete notebook uses, but the state
    is now a continuous position and an action is a displacement rather than one of four moves.
    `sub` sets the resolution of the fine grid used for ground truth: a Dijkstra field on it gives
    the geodesic distance to a goal, which is what "optimal" means once motion is continuous.
    """
    walls = maze_walls(n if n % 2 else n + 1, seed)
    n = walls.shape[0]
    free_cells = np.argwhere(~walls)

    # fine grid of sub-cell centres over free space, for the geodesic field
    fine = np.zeros((n * sub, n * sub), bool)
    for r, c in free_cells:
        fine[r * sub:(r + 1) * sub, c * sub:(c + 1) * sub] = True
    return dict(walls=walls, n=n, free_cells=free_cells, step=step,
                goal_radius=goal_radius, sub=sub, fine=fine)


def is_free(W, pos):
    """Whether a continuous position lies in an open cell."""
    cell = np.floor(pos).astype(int)
    inside = (cell >= 0).all(-1) & (cell < W["n"]).all(-1)
    cell = np.clip(cell, 0, W["n"] - 1)
    return inside & ~W["walls"][cell[..., 0], cell[..., 1]]


def step_positions(W, pos, act):
    """Move, sliding along a wall rather than sticking to it when the full move is blocked."""
    delta = W["step"] * np.clip(act, -1.0, 1.0)
    out = pos.copy()
    for cand in (pos + delta,                                   # full move
                 pos + np.stack([delta[..., 0], np.zeros_like(delta[..., 1])], -1),   # row only
                 pos + np.stack([np.zeros_like(delta[..., 0]), delta[..., 1]], -1)):  # col only
        ok = is_free(W, cand) & ~is_free(W, out) if False else is_free(W, cand)
        take = ok & (out == pos).all(-1)                        # first candidate that works
        out = np.where(take[..., None], cand, out)
    return out


def sample_positions(W, k, rng):
    """Uniform over open space: a uniform cell, then a uniform point inside it."""
    idx = rng.integers(0, len(W["free_cells"]), k)
    return (W["free_cells"][idx] + rng.uniform(0.15, 0.85, (k, 2))).astype(np.float32)


def geodesic_field(W, goal):
    """Distance from every fine-grid node to `goal`, in world units, around the walls.

    Dijkstra on the 8-connected fine grid: this is what a shortest path actually costs when the
    agent can move in any direction, and it is the reference the learned critics are scored on.
    """
    sub, n = W["sub"], W["n"]
    H = n * sub
    cell = 1.0 / sub
    # float64: the heap keys are Python floats, and storing them rounded to float32 makes the
    # "already settled" check fire spuriously, which silently truncates the search
    dist = np.full((H, H), np.inf)
    gr, gc = np.clip((goal * sub).astype(int), 0, H - 1)
    if not W["fine"][gr, gc]:
        return dist
    dist[gr, gc] = 0.0
    heap = [(0.0, gr, gc)]
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr, dc, w in steps:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < H and W["fine"][nr, nc]:
                nd = d + w * cell
                if nd < dist[nr, nc] - 1e-9:
                    dist[nr, nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    return dist


def field_at(W, field, pos):
    """Sample a fine-grid field at continuous positions."""
    H = W["n"] * W["sub"]
    idx = np.clip((pos * W["sub"]).astype(int), 0, H - 1)
    return field[idx[..., 0], idx[..., 1]]


def descent_direction(W, field, pos):
    """The unit direction that most reduces the geodesic field -- the ideal action at `pos`."""
    offs = np.array([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)],
                    np.float32)
    probes = pos[:, None, :] + offs[None] * (1.0 / W["sub"])
    vals = field_at(W, field, probes)
    vals = np.where(np.isfinite(vals), vals, np.inf)
    best = offs[np.argmin(vals, 1)]
    return best / (np.linalg.norm(best, axis=-1, keepdims=True) + 1e-9)
