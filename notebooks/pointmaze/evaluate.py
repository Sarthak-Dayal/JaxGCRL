"""Optimizing a continuous critic (sampled argmax) and scoring what comes out."""

import jax.numpy as jnp
import numpy as np

from .world import descent_direction, field_at, geodesic_field, sample_positions, step_positions


def best_chunk(e, pos, goal):
    """The candidate chunk the critic scores highest -- the continuous stand-in for the argmax."""
    C = jnp.asarray(e["cand"])
    s = jnp.repeat(jnp.asarray(pos)[:, None, :], C.shape[0], 1)
    g = jnp.repeat(jnp.asarray(goal)[:, None, :], C.shape[0], 1)
    qs = e["q"](e["params"], s, g, jnp.broadcast_to(C, (len(pos),) + C.shape))
    qs = jnp.min(qs, -1) if qs.shape[-1] == 2 else qs[..., 0]
    return np.asarray(e["cand"])[np.asarray(jnp.argmax(qs, -1))], np.asarray(qs)


def greedy_action(e, pos, goal):
    """Execute only the first action of the best chunk, then replan."""
    return best_chunk(e, pos, goal)[0][:, :2]


def evaluate(W, e, n_eval=200, seed=0, budget=None, n_probe=400):
    rng = np.random.default_rng(seed)
    budget = budget or int(8 * W["n"] / W["step"])

    starts = sample_positions(W, n_eval, rng)
    goals = sample_positions(W, n_eval, rng)
    pos = starts.copy()
    alive = np.linalg.norm(pos - goals, axis=-1) >= W["goal_radius"]
    visits = []
    for _ in range(budget):
        if not alive.any():
            break
        visits.append(pos[alive].copy())
        pos[alive] = step_positions(W, pos[alive], greedy_action(e, pos[alive], goals[alive]))
        alive &= np.linalg.norm(pos - goals, axis=-1) >= W["goal_radius"]
    success = float((~alive).mean())

    # Alignment with the geodesic descent direction: the continuous analogue of "picks a
    # BFS-optimal move". Cosine of 1 means straight along the shortest path around the walls.
    probe_pos = sample_positions(W, n_probe, rng)
    probe_goals = sample_positions(W, 8, rng)
    cos = []
    for g in probe_goals:
        field = geodesic_field(W, g)
        ideal = descent_direction(W, field, probe_pos)
        act = greedy_action(e, probe_pos, np.repeat(g[None], n_probe, 0))
        act = act / (np.linalg.norm(act, axis=-1, keepdims=True) + 1e-9)
        cos.append((act * ideal).sum(-1))
    return dict(success=success, align=float(np.mean(cos)),
                visits=np.concatenate(visits) if visits else np.zeros((0, 2)))


def random_alignment(W, n_probe=400, seed=0):
    """What a uniformly random action scores on the alignment metric -- the floor for the bars."""
    rng = np.random.default_rng(seed)
    pos = sample_positions(W, n_probe, rng)
    cos = []
    for g in sample_positions(W, 8, rng):
        ideal = descent_direction(W, geodesic_field(W, g), pos)
        act = rng.uniform(-1, 1, (n_probe, 2))
        act /= np.linalg.norm(act, axis=-1, keepdims=True) + 1e-9
        cos.append((act * ideal).sum(-1))
    return float(np.mean(cos))


def action_field(W, e, goal, k=18):
    """The critic's chosen action on a grid of positions, for quiver plots."""
    lo, hi = 0.5, W["n"] - 0.5
    xs = np.linspace(lo, hi, k)
    rr, cc = np.meshgrid(xs, xs, indexing="ij")
    pos = np.stack([rr.ravel(), cc.ravel()], -1).astype(np.float32)
    keep = np.array([bool(x) for x in _free(W, pos)])
    pos = pos[keep]
    act = greedy_action(e, pos, np.repeat(np.asarray(goal)[None], len(pos), 0))
    return pos, act / (np.linalg.norm(act, axis=-1, keepdims=True) + 1e-9)


def ideal_field(W, goal, k=18):
    lo, hi = 0.5, W["n"] - 0.5
    xs = np.linspace(lo, hi, k)
    rr, cc = np.meshgrid(xs, xs, indexing="ij")
    pos = np.stack([rr.ravel(), cc.ravel()], -1).astype(np.float32)
    pos = pos[np.array([bool(x) for x in _free(W, pos)])]
    return pos, descent_direction(W, geodesic_field(W, goal), pos)


def value_grid(W, e, goal, k=40):
    """max over candidate chunks at each position, as an image."""
    xs = np.linspace(0, W["n"], k)
    rr, cc = np.meshgrid(xs, xs, indexing="ij")
    pos = np.stack([rr.ravel(), cc.ravel()], -1).astype(np.float32)
    _, qs = best_chunk(e, pos, np.repeat(np.asarray(goal)[None], len(pos), 0))
    img = qs.max(-1).reshape(k, k)
    free = np.array([bool(x) for x in _free(W, pos)]).reshape(k, k)
    return np.where(free, img, np.nan)


def geodesic_grid(W, goal, k=40):
    xs = np.linspace(0, W["n"], k)
    rr, cc = np.meshgrid(xs, xs, indexing="ij")
    pos = np.stack([rr.ravel(), cc.ravel()], -1).astype(np.float32)
    d = field_at(W, geodesic_field(W, goal), pos).reshape(k, k)
    return np.where(np.isfinite(d), d, np.nan)


def _free(W, pos):
    from .world import is_free
    return is_free(W, pos)
