"""Optimizing a continuous critic (sampled argmax) and scoring what comes out."""

import jax
import jax.numpy as jnp
import numpy as np

from .world import descent_direction, geodesic, sample_positions, step_positions

GOOD_FRACTION = 0.5     # a pick counts as correct when it makes at least this share of the
                        # best progress any candidate would have made


def _cpu(W, e):
    """The critic's evaluation-side functions, jitted on the CPU and cached on the critic.

    Evaluation is hundreds of calls on a few hundred positions each: on a GPU that is
    kernel-launch overhead and nothing else. Compiled once per critic (and per input shape),
    with a CPU copy of the parameters. The greedy rollout is a single scan, with the point-mass
    dynamics (wall sliding included) written in jnp to match `world.step_positions`.
    """
    if "_cpu" not in e:
        cpu = jax.devices("cpu")[0]
        C, params = jax.device_put(e["cand"], cpu), jax.device_put(e["params"], cpu)
        walls = jax.device_put(W["walls"], cpu)
        n, step, radius = W["n"], W["step"], W["goal_radius"]

        def free(pos):
            cell = jnp.floor(pos).astype(jnp.int32)
            inside = (cell >= 0).all(-1) & (cell < n).all(-1)
            cell = jnp.clip(cell, 0, n - 1)
            return inside & ~walls[cell[..., 0], cell[..., 1]]

        def move(pos, act):
            delta = step * jnp.clip(act, -1.0, 1.0)
            out = pos
            for cand in (pos + delta,
                         pos + delta * jnp.array([1.0, 0.0]),
                         pos + delta * jnp.array([0.0, 1.0])):
                take = free(cand) & (out == pos).all(-1)
                out = jnp.where(take[..., None], cand, out)
            return out

        def q_all(pos, goal):
            """(len(pos), n_cand): the critic at every candidate chunk."""
            s = jnp.repeat(pos[:, None, :], C.shape[0], 1)
            g = jnp.repeat(goal[:, None, :], C.shape[0], 1)
            qs = e["q"](params, s, g, jnp.broadcast_to(C, (pos.shape[0],) + C.shape))
            return jnp.min(qs, -1) if qs.shape[-1] == 2 else qs[..., 0]

        def greedy(pos, goal):
            """Execute only the first action of the best chunk, then replan."""
            return C[jnp.argmax(q_all(pos, goal), -1), :2]

        def rollout(pos, goal, budget):
            def body(carry, _):
                pos, alive, visits = carry
                visits = visits + alive[:, None]
                pos = jnp.where(alive[:, None], move(pos, greedy(pos, goal)), pos)
                alive = alive & (jnp.linalg.norm(pos - goal, axis=-1) >= radius)
                return (pos, alive, visits), pos

            alive = jnp.linalg.norm(pos - goal, axis=-1) >= radius
            (pos, alive, _), path = jax.lax.scan(body, (pos, alive, jnp.zeros((len(pos), 1))),
                                                 None, length=budget)
            return ~alive, path

        e["_cpu"] = dict(q_all=jax.jit(q_all), greedy=jax.jit(greedy),
                         rollout=jax.jit(rollout, static_argnames="budget"), device=cpu)
    return e["_cpu"]


def _to(f, *xs):
    return [jax.device_put(np.asarray(x, np.float32), f["device"]) for x in xs]


def q_over_candidates(W, e, pos, goal):
    f = _cpu(W, e)
    return np.asarray(f["q_all"](*_to(f, pos, goal)))


def greedy_action(W, e, pos, goal):
    f = _cpu(W, e)
    return np.asarray(f["greedy"](*_to(f, pos, goal)))


def rollouts(W, e, pos, goal, budget):
    """Greedy rollouts from `pos` to `goal`: (arrived mask, positions over time)."""
    f = _cpu(W, e)
    arrived, path = f["rollout"](*_to(f, pos, goal), budget)
    return np.asarray(arrived), np.asarray(path)


def progress(W, pos, goal, act):
    """How much closer to the goal, along the geodesic, one step of `act` from `pos` gets."""
    return geodesic(W, pos, goal) - geodesic(W, step_positions(W, pos, act), goal)


def good_pick(W, e, pos, goal, act):
    """Whether `act` makes at least GOOD_FRACTION of the best progress any candidate would make
    from `pos` -- the continuous stand-in for "picks a BFS-optimal move". Relative to the pool
    rather than to an ideal direction, because that pool is what the critic chooses among, and
    it handles sliding along walls the way the dynamics do."""
    cand = np.asarray(e["cand"])[:, :2]
    best = np.max([progress(W, pos, goal, np.repeat(c[None], len(pos), 0)) for c in cand], 0)
    return progress(W, pos, goal, act) >= GOOD_FRACTION * best


def evaluate(W, e, n_eval=200, seed=0, budget=None, n_probe=400, n_goals=8):
    """Two scores. `agree` is the share of (position, goal) probes where the greedy action is a
    good pick (see `good_pick`). `success` is the share of greedy rollouts that enter the goal
    radius within the budget."""
    rng = np.random.default_rng(seed)
    budget = budget or int(8 * W["n"] / W["step"])

    arrived, _ = rollouts(W, e, sample_positions(W, n_eval, rng), sample_positions(W, n_eval, rng),
                          budget)
    pos = sample_positions(W, n_probe, rng)
    good = []
    for g in sample_positions(W, n_goals, rng):
        goal = np.repeat(g[None], n_probe, 0)
        good.append(good_pick(W, e, pos, goal, greedy_action(W, e, pos, goal)))
    return dict(success=float(arrived.mean()), agree=float(np.concatenate(good).mean()))


def random_floor(W, cand, n_probe=400, seed=0):
    """What a uniformly random candidate scores on `agree` -- the floor for the charts."""
    rng = np.random.default_rng(seed)
    e = dict(cand=cand)
    pos = sample_positions(W, n_probe, rng)
    good = []
    for g in sample_positions(W, 8, rng):
        goal = np.repeat(g[None], n_probe, 0)
        act = np.asarray(cand)[rng.integers(0, len(cand), n_probe), :2]
        good.append(good_pick(W, e, pos, goal, act))
    return float(np.concatenate(good).mean())


def cell_grid(W):
    """One probe position per free cell: its centre. Where the arrows are drawn."""
    return (W["free_cells"] + 0.5).astype(np.float32)


def action_field(W, e, goal):
    """The critic's chosen action at every cell centre, as unit vectors, and whether each is a
    good pick."""
    pos = cell_grid(W)
    goal = np.repeat(np.asarray(goal, np.float32)[None], len(pos), 0)
    act = greedy_action(W, e, pos, goal)
    good = good_pick(W, e, pos, goal, act)
    return act / (np.linalg.norm(act, axis=-1, keepdims=True) + 1e-9), good


def ideal_field(W, goal):
    pos = cell_grid(W)
    return descent_direction(W, pos, np.repeat(np.asarray(goal, np.float32)[None], len(pos), 0))


def fine_positions(W):
    """The centres of the fine-grid nodes, as an (H, H, 2) array, plus the free mask."""
    H = W["n"] * W["sub"]
    xs = (np.arange(H) + 0.5) / W["sub"]
    rr, cc = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([rr, cc], -1).astype(np.float32), W["fine"]


def value_grid(W, e, goal):
    """max over candidate chunks at every fine-grid node, as an image (NaN in walls)."""
    pos, free = fine_positions(W)
    flat = pos.reshape(-1, 2)
    qs = q_over_candidates(W, e, flat, np.repeat(np.asarray(goal, np.float32)[None], len(flat), 0))
    return np.where(free, qs.max(-1).reshape(free.shape), np.nan)


def geodesic_grid(W, goal):
    pos, free = fine_positions(W)
    flat = pos.reshape(-1, 2)
    d = geodesic(W, flat, np.repeat(np.asarray(goal, np.float32)[None], len(flat), 0))
    return np.where(free, d.reshape(free.shape), np.nan)
