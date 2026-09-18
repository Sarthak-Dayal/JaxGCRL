"""Data distributions of controllable coverage, from random walk to full.

The critics elsewhere in this package are trained on uniform random walks, which in any maze-like
world means the relabeled goal is always near the state it is paired with: the walk is diffusive,
so it never travels far enough for a distant (state, goal) pair to exist in the data. Evaluation,
though, samples goals uniformly. This module makes that gap a dial rather than a fixed defect.

The dial is an outward-biased policy. At each step it takes the action that most increases the BFS
distance from the trajectory's start with probability `gamma`, and acts uniformly otherwise. Runs
of outward moves are then geometric with mean `1 / (1 - gamma)`, so parameterizing by an effective
distance D and setting gamma = 1 - 1/D means "push outward about D steps before collapsing back to
a random walk". D = 1 is exactly the uniform random walk; large D reaches the far end of the maze.

`uniform_batches` is the idealized ceiling: state, action and goal all drawn uniformly, so every
(state, goal) distance is represented in proportion to how many such pairs exist.
"""

import numpy as np

from .data import potential
from .world import GAMMA, N_ACT


def collect_outward(W, start, D=1.0, n_traj=400, T=None, seed=0):
    """Random walks biased outward from a moving anchor, every one beginning at cell `start`.

    D is the effective excursion length: gamma = 1 - 1/D is the per-step probability of taking the
    action that most increases the BFS distance from the anchor. Taking a random action instead
    re-anchors to the current cell, so directed runs are geometric with mean D and the walk starts
    a fresh excursion afterwards. D = 1 gives gamma = 0, the plain random walk.

    The anchor has to move. Biasing outward from a *fixed* start makes the walk sprint to the far
    wall and stall there, and since the relabeled goal is drawn from the trajectory's future, a
    stalled tail produces goals sitting right on top of their state -- the opposite of coverage.
    """
    if T is None:
        _d = W["dist"][np.isfinite(W["dist"])].mean()
        T = int(np.clip(2 * _d**2, 40, 2000))
    gamma = 0.0 if D <= 1 else 1.0 - 1.0 / D

    rng = np.random.default_rng(seed)
    S = np.zeros((n_traj, T + 1), np.int32)
    A = np.zeros((n_traj, T), np.int32)
    S[:, 0] = start
    anchor = S[:, 0].copy()
    rows = np.arange(n_traj)

    for t in range(T):
        here = S[:, t]
        nxt = W["nxt"][here]                              # (n_traj, 4) candidate next cells
        d_anchor = W["dist"][anchor]
        gain = d_anchor[rows[:, None], nxt] + rng.uniform(0, 1e-3, nxt.shape)
        go_out = rng.random(n_traj) < gamma
        A[:, t] = np.where(go_out, np.argmax(gain, axis=1), rng.integers(0, N_ACT, n_traj))

        # A random step restarts the excursion; so does running out of room. Without the second
        # case a long excursion ends up ping-ponging against the far wall, which pulls coverage
        # back down at exactly the settings meant to maximize it.
        stuck = gain.max(1) <= d_anchor[rows, here] + 1e-3
        anchor = np.where(go_out & ~stuck, anchor, here)
        S[:, t + 1] = W["nxt"][here, A[:, t]]
    return S, A


def uniform_batches(W, h, n_batches, batch=256, seed=1, shaping=0.0):
    """The ceiling: state, action sequence and goal all uniform, so no distance is under-sampled.

    Built directly rather than by relabeling a trajectory -- there is no trajectory. The reward and
    discount follow the same absorbing-goal convention as `data.make_batches`, so the two are
    interchangeable as far as the objectives are concerned.
    """
    rng = np.random.default_rng(seed)
    shape = (n_batches, batch)
    state = rng.integers(0, W["N"], shape).astype(np.int32)
    goal = rng.integers(0, W["N"], shape).astype(np.int32)
    seq = rng.integers(0, N_ACT, shape + (h,)).astype(np.int32)

    gam = W["gamma"]
    reward = np.zeros(shape, np.float32)
    discount = np.full(shape, gam**h, np.float32)
    cur, done = state.copy(), np.zeros(shape, bool)
    for k in range(h):
        cur = W["nxt"][cur, seq[..., k]]
        hit = (~done) & (cur == goal)
        reward[hit] = gam**k
        discount[hit] = 0.0
        done |= hit

    if shaping:
        reward = reward + discount * potential(W, cur, goal, shaping) \
            - potential(W, state, goal, shaping)
    return dict(state=state, goal=goal, seq=seq.reshape(shape + (h,)).reshape(*shape, h),
                reward=reward, discount=discount, next_state=cur)


def pair_distances(W, batches, n=20000):
    """The distribution of BFS distances between the (state, goal) pairs a dataset actually
    contains -- the thing coverage is really about."""
    s = np.asarray(batches["state"]).ravel()[:n]
    g = np.asarray(batches["goal"]).ravel()[:n]
    d = W["dist"][g, s]
    return d[np.isfinite(d)]


# The critics this study compares: three factorizations, and for each the objectives it admits.
#
# InfoNCE only identifies the critic up to an offset along the axis it does *not* normalize
# over, and a greedy policy is a comparison across actions at a fixed goal. Forward InfoNCE on
# sa_g -- CRL's default -- normalizes each (s, a) row over goals, leaving a free offset per
# (s, a): exactly the comparison the argmax needs (0.81 on D=4 and 0.32 on D=64 here, against a
# regression ceiling of 0.91 and 0.90). Backward InfoNCE normalizes down the goal's column, over
# (s, a), and recovers 0.87 and 0.71, so that is the direction sa_g gets. sg_a has no such choice
# to make: its columns are action sequences, so either direction compares across actions, and
# it sits at the ceiling (0.92-0.96) in both. Binary NCE scores pairs independently and has no
# offset at all. The monolithic critic has no two-tower structure and so no contrastive form.
CRITICS = [
    ("monolithic_td", "monolithic", "td"),
    ("sa_g_td", "sa_g", "td"),
    ("sg_a_td", "sg_a", "td"),
    ("sa_g_infonce", "sa_g", "bwd_infonce"),        # normalized over (s, a)
    ("sg_a_infonce", "sg_a", "infonce"),            # normalized over action sequences
    ("sa_g_binary_nce", "sa_g", "binary_nce"),
    ("sg_a_binary_nce", "sg_a", "binary_nce"),
    # The control: the same three architectures regressed directly on the optimal Q. Same data,
    # same capacity, perfect targets -- so whatever these cannot reach is a limit of the coverage
    # or the network, and the gap below them is what the RL objective costs.
    ("monolithic_mse", "monolithic", "mse"),
    ("sa_g_mse", "sa_g", "mse"),
    ("sg_a_mse", "sg_a", "mse"),
]


def train_suite(W, h, batches, td_batches=None, seed=0, kind="sac", shaping=0.0, lr=3e-4,
                width=128, repr_dim=32):
    """Train every critic in CRITICS on one dataset.

    `batches` is an already-built batch dict whose leading axis is the number of gradient steps;
    the contrastive critics train on it. The TD and regression critics train on `td_batches`
    (same data, hindsight goals drawn uniformly from the future) when given, else on `batches`.
    `width` is every MLP's hidden width and `repr_dim` the two-tower embedding size.
    """
    import jax

    from .critics import monolithic, sa_g_bilinear, sg_a_bilinear
    from .data import add_optimal_targets
    from .training import train_contrastive, train_regression, train_td
    from .world import all_chunks

    fd = W["feats"].shape[1]
    chunks = all_chunks(h)
    td_batches = add_optimal_targets(W, batches if td_batches is None else td_batches, shaping)

    out = {}
    for i, (name, factor, objective) in enumerate(CRITICS):
        key = jax.random.PRNGKey(seed + i)
        # A distance energy cannot represent a positive target, so the regression and TD critics
        # take `dot` while the contrastive ones keep `norm`, as JaxGCRL does.
        energy = "norm" if objective in ("infonce", "bwd_infonce", "sym_infonce", "binary_nce") else "dot"
        build = {"monolithic": lambda: monolithic(h, fd, width),
                 "sa_g": lambda: sa_g_bilinear(h, fd, repr_dim, energy, width),
                 "sg_a": lambda: sg_a_bilinear(h, fd, repr_dim, energy, width)}[factor]()
        if objective == "td":
            p, q, losses, accs = train_td(build, td_batches, W["feats"], chunks, kind, key,
                                          lr=lr)
        elif objective == "mse":
            p, q, losses, accs = train_regression(build, td_batches, W["feats"], key, lr=lr)
        else:
            p, q, losses, accs = train_contrastive(build, batches, W["feats"], key, lr=lr,
                                                   loss_fn=objective)
        out[name] = dict(params=p, q=q, chunks=chunks, losses=losses, accs=accs,
                         objective=objective,
                         n_params=sum(x.size for x in jax.tree_util.tree_leaves(p)))
        yield name, out
