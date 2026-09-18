"""A continuous point mass in a maze: positions in R^2, actions in [-1, 1]^2."""

from collections import deque

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from gridworld.world import maze_walls


def braid_walls(n, seed=0, fraction=0.3):
    """A perfect maze with a fraction of its dead ends opened up, so it has cycles.

    Every wall that separates two corridor cells is a candidate; opening one joins two branches
    of the tree and creates a loop. With loops there are alternative routes between most pairs of
    cells, of different lengths and widths -- the case where one step of policy improvement over
    a random walk is *not* the optimal action.
    """
    walls = maze_walls(n, seed)
    n = walls.shape[0]
    rng = np.random.default_rng(seed + 1)
    cand = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
            if walls[r, c] and (r + c) % 2 == 1
            and ((not walls[r - 1, c] and not walls[r + 1, c])
                 or (not walls[r, c - 1] and not walls[r, c + 1]))]
    for i in rng.permutation(len(cand))[:int(fraction * len(cand))]:
        walls[cand[i]] = False
    return walls


def room_walls(n, seed=0, rooms=3):
    """A grid of rooms with one door in every shared wall, at a random position.

    Open space inside each room, so the value function has to be smooth there and jump at the
    walls; many loops, since every 2x2 block of rooms is a cycle; and a horizon set by the number
    of doors between two points rather than by a winding corridor.
    """
    rng = np.random.default_rng(seed)
    walls = np.ones((n, n), bool)
    edges = np.linspace(0, n - 1, rooms + 1).round().astype(int)  # room boundaries, inclusive
    for i in range(rooms):
        for j in range(rooms):
            walls[edges[i] + 1:edges[i + 1], edges[j] + 1:edges[j + 1]] = False
    for i in range(rooms):
        for j in range(rooms):
            if j + 1 < rooms:                                     # door in the wall to the right
                walls[rng.integers(edges[i] + 1, edges[i + 1]), edges[j + 1]] = False
            if i + 1 < rooms:                                     # door in the wall below
                walls[edges[i + 1], rng.integers(edges[j] + 1, edges[j + 1])] = False
    return walls


def build_world(kind="maze", n=13, seed=0, step=0.5, goal_radius=0.6, sub=3, braid=0.3,
                half_life=1.0):
    """A point-mass maze of one of three topologies.

    `maze` is a perfect maze (a tree: one path between any two points), `braid` the same maze
    with `braid` of its dead ends opened into loops, `rooms` a 3x3 grid of rooms joined by doors.
    The state is a continuous position and an action a displacement of up to `step` per axis.
    `sub` sets the resolution of the fine grid used for ground truth: a geodesic distance on it,
    around the walls, is what "optimal" means once motion is continuous.
    """
    n = n if n % 2 else n + 1
    walls = {"maze": lambda: maze_walls(n, seed),
             "braid": lambda: braid_walls(n, seed, braid),
             "rooms": lambda: room_walls(n, seed)}[kind]()
    free_cells = np.argwhere(~walls)
    N = len(free_cells)
    index = {tuple(rc): i for i, rc in enumerate(free_cells)}

    # the coarse cell graph and its BFS distances: what the outward-biased collector steers by
    cell_nxt = np.zeros((N, 4), np.int32)
    for i, (r, c) in enumerate(free_cells):
        for a, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            cell_nxt[i, a] = index.get((r + dr, c + dc), i)
    cell_dist = np.full((N, N), np.inf, np.float32)
    for g in range(N):
        cell_dist[g, g] = 0
        q = deque([g])
        while q:
            u = q.popleft()
            for v in cell_nxt[u]:
                if cell_dist[g, v] == np.inf:
                    cell_dist[g, v] = cell_dist[g, u] + 1
                    q.append(v)

    # fine grid of sub-cell nodes over free space, for the geodesic field
    fine = np.zeros((n * sub, n * sub), bool)
    for r, c in free_cells:
        fine[r * sub:(r + 1) * sub, c * sub:(c + 1) * sub] = True
    W = dict(walls=walls, n=n, free_cells=free_cells, N=N, cell_nxt=cell_nxt,
             cell_dist=cell_dist, step=step, goal_radius=goal_radius, sub=sub, fine=fine)
    # gamma scales with the world, as in the gridworld: values halve over `half_life` mean
    # geodesic paths, so they stay spread over the distances that exist rather than vanishing
    # at the far end of a long maze.
    d = fine_distances(W)
    W["gamma"] = float(0.5 ** (step / max(half_life * d[np.isfinite(d)].mean(), step)))
    return W


def cell_of(W, pos):
    """The free-cell index of each continuous position (-1 if inside a wall)."""
    cell = np.clip(np.floor(pos).astype(int), 0, W["n"] - 1)
    lookup = np.full(W["walls"].shape, -1, np.int32)
    lookup[W["free_cells"][:, 0], W["free_cells"][:, 1]] = np.arange(W["N"])
    return lookup[cell[..., 0], cell[..., 1]]


def cell_centre(W, cell):
    return (W["free_cells"][cell] + 0.5).astype(np.float32)


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
        take = is_free(W, cand) & (out == pos).all(-1)          # first candidate that works
        out = np.where(take[..., None], cand, out)
    return out


def sample_positions(W, k, rng):
    """Uniform over open space: a uniform cell, then a uniform point inside it."""
    idx = rng.integers(0, len(W["free_cells"]), k)
    return (W["free_cells"][idx] + rng.uniform(0.15, 0.85, (k, 2))).astype(np.float32)


def fine_distances(W):
    """Geodesic distance between every pair of fine-grid nodes, in world units, around the
    walls: Dijkstra on the 8-connected grid. Computed once and cached on the world; it is what
    the oracle Q, the ideal direction and the alignment score are all read from."""
    if "fine_dist" in W:
        return W["fine_dist"]
    sub, H = W["sub"], W["n"] * W["sub"]
    node = np.full((H, H), -1)
    free = np.argwhere(W["fine"])
    node[free[:, 0], free[:, 1]] = np.arange(len(free))
    rows, cols, wts = [], [], []
    for dr, dc, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
        r, c = free[:, 0] + dr, free[:, 1] + dc
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < H)
        ok[ok] &= node[r[ok], c[ok]] >= 0
        rows.append(node[free[ok, 0], free[ok, 1]]); cols.append(node[r[ok], c[ok]]); wts.append(np.full(ok.sum(), w / sub))
    graph = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(free), len(free)))
    W["fine_dist"] = shortest_path(graph, directed=False).astype(np.float32)
    W["fine_node"] = node
    return W["fine_dist"]


def fine_node(W, pos):
    """The fine-grid node of each continuous position (-1 inside a wall)."""
    fine_distances(W)
    H = W["n"] * W["sub"]
    idx = np.clip((np.asarray(pos) * W["sub"]).astype(int), 0, H - 1)
    return W["fine_node"][idx[..., 0], idx[..., 1]]


def geodesic(W, pos, goal):
    """Geodesic distance from `pos` to `goal`, elementwise; inf where either is in a wall."""
    a, b = fine_node(W, pos), fine_node(W, goal)
    d = fine_distances(W)[np.clip(a, 0, None), np.clip(b, 0, None)]
    return np.where((a >= 0) & (b >= 0), d, np.inf)


def descent_direction(W, pos, goal):
    """The unit direction that most reduces the geodesic distance to `goal` -- the ideal action."""
    offs = np.array([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)],
                    np.float32)
    probes = pos[:, None, :] + offs[None] * (1.0 / W["sub"])
    vals = geodesic(W, probes, np.repeat(goal[:, None, :], len(offs), 1))
    best = offs[np.argmin(vals, 1)]
    return best / (np.linalg.norm(best, axis=-1, keepdims=True) + 1e-9)
