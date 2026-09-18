"""Optimizing a critic (argmax over chunks) and scoring what comes out."""

import jax
import jax.numpy as jnp
import numpy as np

from .data import optimal_q
from .world import GAMMA


def _cpu(W, e):
    """The critic's evaluation-side functions, jitted on the CPU and cached on the critic.

    Evaluation is hundreds of calls on a few hundred states each: on a GPU that is kernel-launch
    overhead and nothing else, and the CPU does it in a fraction of the time regardless of where
    the critic trained. Compiled once per critic (and per input shape), with a CPU copy of the
    parameters, so nothing here is traced or transferred per call.
    """
    if "_cpu" not in e:
        cpu = jax.devices("cpu")[0]
        F, C, NXT, params = (jax.device_put(x, cpu)
                             for x in (W["feats"], e["chunks"], W["nxt"], e["params"]))

        def q_all(states, goals):
            """(len(states), n_chunks): the critic at every action sequence."""
            sf = jnp.repeat(F[states][:, None, :], C.shape[0], 1)
            gf = jnp.repeat(F[goals][:, None, :], C.shape[0], 1)
            qs = e["q"](params, sf, gf, jnp.broadcast_to(C, (states.shape[0],) + C.shape))
            return jnp.min(qs, -1) if qs.shape[-1] == 2 else qs[..., 0]

        def greedy(states, goals):
            """Optimizing the critic: score every chunk, take the best one's first action."""
            return C[jnp.argmax(q_all(states, goals), -1), 0]

        def rollout(s, goals, budget):
            """Greedy rollouts for `budget` steps. Returns visits per cell (counted before each
            move, walkers that have arrived stop counting) and which walkers arrived."""
            def step(carry, _):
                s, alive, visits = carry
                visits = visits.at[s].add(alive)
                s = jnp.where(alive, NXT[s, greedy(s, goals)], s)
                return (s, alive & (s != goals), visits), None

            init = (s, s != goals, jnp.zeros(W["N"], jnp.int32))
            (s, alive, visits), _ = jax.lax.scan(step, init, None, length=budget)
            return visits, ~alive

        e["_cpu"] = dict(q_all=jax.jit(q_all), greedy=jax.jit(greedy),
                         rollout=jax.jit(rollout, static_argnames="budget"), device=cpu)
    return e["_cpu"]


def q_over_chunks(W, e, states, goals):
    """(len(states), n_chunks) scores -- the critic evaluated at every action sequence."""
    f = _cpu(W, e)
    return f["q_all"](jax.device_put(states, f["device"]), jax.device_put(goals, f["device"]))


def greedy_action(W, e, states, goals):
    """Optimizing the critic: score every chunk, take the best one's first action."""
    f = _cpu(W, e)
    return np.asarray(f["greedy"](jax.device_put(states, f["device"]),
                                  jax.device_put(goals, f["device"])))


def _rollout(W, e, s, goals, budget):
    f = _cpu(W, e)
    visits, arrived = f["rollout"](jax.device_put(s, f["device"]),
                                   jax.device_put(goals, f["device"]), budget)
    return np.asarray(visits), np.asarray(arrived)


def evaluate(W, e, n_eval=300, n_probe=400, seed=0, budget=None, shaping=0.0):
    rng = np.random.default_rng(seed)
    # A maze needs far more steps than a room: scale with the longest shortest-path, not the side.
    budget = budget or int(2 * W["dist"][np.isfinite(W["dist"])].max())
    N = W["N"]

    visited, arrived = _rollout(W, e, rng.integers(0, N, n_eval), rng.integers(0, N, n_eval),
                                budget)
    success = float(arrived.mean())

    # does the critic prefer a genuinely optimal move? BFS says which moves are optimal.
    gs, gg = rng.integers(0, N, n_probe), rng.integers(0, N, n_probe)
    act = greedy_action(W, e, gs, gg)
    d_after = W["dist"][gg[:, None], W["nxt"][gs]]                  # (probe, 4)
    agree = float(np.mean(d_after[np.arange(n_probe), act] <= d_after.min(1) + 1e-6))

    # rank agreement with the optimal Q over the chunks it actually scores
    qs = np.asarray(q_over_chunks(W, e, gs, gg))
    truth = np.stack([optimal_q(W, gs, gg, np.repeat(c[None], n_probe, 0), shaping)
                      for c in e["chunks"]], 1)
    rank = float(np.mean([_spearman(qs[i], truth[i]) for i in range(n_probe)])) if qs.shape[1] > 1 else float("nan")

    return dict(success=success, coverage=float((visited > 0).sum() / N), agree=agree,
                rank=rank, visited=visited)


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def value_map(W, e, goal):
    """V(s, g) = max over chunks, for every free cell."""
    states = np.arange(W["N"])
    qs = np.asarray(q_over_chunks(W, e, states, np.full(W["N"], goal)))
    return qs.max(-1)


def to_grid(W, vals):
    img = np.full(W["walls"].shape, np.nan)
    img[W["cells"][:, 0], W["cells"][:, 1]] = vals
    return img


def greedy_field(W, e, goal):
    """The critic's chosen first action at every free cell, for one fixed goal."""
    return greedy_action(W, e, np.arange(W["N"]), np.full(W["N"], goal))


def optimal_field(W, goal):
    """(N, 4) mask of the BFS-optimal actions at every cell. Ties are all marked optimal."""
    d_after = W["dist"][goal][W["nxt"]]
    return d_after <= d_after.min(1, keepdims=True) + 1e-6


def rollouts_to_goal(W, e, goal, budget=None):
    """Greedy rollouts from every free cell to one goal.

    Returns the visit counts and which starts arrived -- the state-visitation distribution the
    trained critic actually induces, rather than coverage aggregated over random goals.
    """
    budget = budget or int(2 * W["dist"][np.isfinite(W["dist"])].max())
    return _rollout(W, e, np.arange(W["N"]), np.full(W["N"], goal), budget)


def data_coverage(W, S, n_probe=2000, seed=0):
    """How far the collected walks travel, against how far the evaluation goals are.

    If the walks do not reach as far as the goals, no critic can rank those goals -- the honest
    reading of a flat comparison is then "not enough data", not "these objectives are equivalent".
    """
    rng = np.random.default_rng(seed)
    walked = W["dist"][S[:, 0], S[:, -1]]
    gs, gg = rng.integers(0, W["N"], n_probe), rng.integers(0, W["N"], n_probe)
    d_after = W["dist"][gg[:, None], W["nxt"][gs]]
    return dict(walk_median=float(np.median(walked)),
                walk_p90=float(np.percentile(walked, 90)),
                goal_mean=float(W["dist"][np.isfinite(W["dist"])].mean()),
                chance=float((d_after <= d_after.min(1, keepdims=True) + 1e-6).mean()))


def agree_by_distance(W, e, edges=(1, 2, 4, 8, 16, 64), n_probe=4000, seed=0):
    """Optimal-move agreement split by how far the goal is.

    The aggregate number hides the thing coverage actually changes. A dataset of near pairs can
    teach a critic to be right near the goal and wrong far from it, and a dataset spread over all
    distances can do the reverse -- while both average out to something unremarkable.
    """
    rng = np.random.default_rng(seed)
    gs, gg = rng.integers(0, W["N"], n_probe), rng.integers(0, W["N"], n_probe)
    act = greedy_action(W, e, gs, gg)
    d_after = W["dist"][gg[:, None], W["nxt"][gs]]
    ok = d_after[np.arange(n_probe), act] <= d_after.min(1) + 1e-6
    d = W["dist"][gg, gs]

    # A random action's hit rate varies with distance -- far from the goal several moves are
    # tied-optimal, so "agreement" drifts up with distance for reasons that have nothing to do
    # with the critic. Carry the floor per bin so the numbers can be read.
    chance = (d_after <= d_after.min(1, keepdims=True) + 1e-6).mean(1)

    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        out.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                        agree=float(ok[m].mean()) if m.any() else float("nan"),
                        chance=float(chance[m].mean()) if m.any() else float("nan")))
    return out
