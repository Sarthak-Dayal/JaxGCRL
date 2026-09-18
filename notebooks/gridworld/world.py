"""The gridworld itself: topology, dynamics, and BFS ground truth."""

from collections import deque

import numpy as np


ACTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # up, down, left, right
N_ACT = len(ACTIONS)
GAMMA = 0.9


def maze_walls(n, seed=0):
    """A perfect maze by recursive backtracking: exactly one path between any two cells.

    Corridors sit on odd coordinates and the wall between two corridor cells is carved when the
    maze visits it, so `n` must be odd. Long winding horizons and dead ends are the point -- the
    value function has to route around obstacles rather than flow smoothly toward the goal.
    """
    rng = np.random.default_rng(seed)
    walls = np.ones((n, n), bool)
    walls[1, 1] = False
    stack = [(1, 1)]
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
                if 0 < r + dr < n - 1 and 0 < c + dc < n - 1 and walls[r + dr, c + dc]]
        if not nbrs:
            stack.pop()
            continue
        nr, nc = nbrs[rng.integers(len(nbrs))]
        walls[(r + nr) // 2, (c + nc) // 2] = False
        walls[nr, nc] = False
        stack.append((nr, nc))
    return walls


def build_world(kind="four rooms", n=11, seed=0, features="coords", gamma=None):
    """Walls, the free-cell index, the transition table, and BFS distances between all cells."""
    if kind == "maze":
        walls = maze_walls(n if n % 2 else n + 1, seed)
        n = walls.shape[0]
    else:
        walls = np.zeros((n, n), bool)
    mid = n // 2
    if kind in ("four rooms", "two rooms"):
        walls[mid, :] = True
        walls[mid, n // 4] = False
        walls[mid, 3 * n // 4] = False
    if kind == "four rooms":
        walls[:, mid] = True
        walls[n // 4, mid] = False
        walls[3 * n // 4, mid] = False
        walls[mid, mid] = True

    cells = [(r, c) for r in range(n) for c in range(n) if not walls[r, c]]
    index = {rc: i for i, rc in enumerate(cells)}
    N = len(cells)

    nxt = np.zeros((N, N_ACT), np.int32)
    for i, (r, c) in enumerate(cells):
        for a, (dr, dc) in enumerate(ACTIONS):
            nxt[i, a] = index.get((r + dr, c + dc), i)  # walking into a wall is a no-op

    dist = np.full((N, N), np.inf, np.float32)          # dist[goal, state]
    for g in range(N):
        dist[g, g] = 0
        q = deque([g])
        while q:
            u = q.popleft()
            for v in nxt[u]:
                if dist[g, v] == np.inf:
                    dist[g, v] = dist[g, u] + 1
                    q.append(v)

    cells = np.array(cells)
    # "coords" is the interesting case -- the critic has to infer the topology from position, and
    # in a maze two cells one wall apart look adjacent. "onehot" is the tabular control: the critic
    # can represent any function of the cell, so what is left is purely the learning objective.
    feats = (np.eye(N, dtype=np.float32) if features == "onehot"
             else (cells / (n - 1)).astype(np.float32))

    # gamma has to scale with the world. At 0.9 a 52-step maze has Q* = 0.9**52 = 0.004 and the
    # gap between a good action and a bad one is ~1e-4: nothing to regress on, nothing to rank,
    # and every critic -- including one handed the exact answer -- sits at chance. Default to a
    # half-life of one mean shortest-path, so values stay spread over the distances that exist.
    finite = dist[np.isfinite(dist)]
    if gamma is None:
        gamma = float(0.5 ** (1.0 / max(finite.mean(), 1.0)))
    return dict(walls=walls, cells=cells, n=n, N=N, nxt=nxt, dist=dist, feats=feats, gamma=gamma)


def all_chunks(n):
    """Every action sequence of length n, as (4^n, n)."""
    return np.array(np.meshgrid(*[np.arange(N_ACT)] * n, indexing="ij")).reshape(n, -1).T.astype(np.int32)
