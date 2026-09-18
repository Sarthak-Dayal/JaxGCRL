"""Data distributions of controllable coverage in the continuous maze, from random to full.

The continuous twin of `gridworld/coverage.py`. The walker carries a heading and moves along it
at a random speed; the heading drifts a little each step and is re-drawn with probability
`1 / D`, or on hitting a wall. Every point knows the direction that leads back toward the start
(the geodesic descent toward it), and with probability `1 - 1/D` a voluntary turn takes its new
heading from the half-plane facing *away* from that direction. So D = 1 is a plain random walk
that turns every step, and large D is long runs that, when they do turn, keep heading out.
Nothing is steered toward anything; the walk only refuses the way home. A turn forced by a wall
is unconstrained: refusing the way home at a dead end means vibrating in it until the trajectory
ends, and a stalled tail relabels goals on top of their states. The speed is random so that the actions in the
data fill the box the critics are later maximized over: at full speed only, every action sits
on the box's boundary, a critic's action tower never sees the interior, and its argmax lands on
extrapolated interior actions that barely move. `uniform` draws state, action and goal
independently over free space.
"""

import numpy as np

from .world import cell_centre, descent_direction, sample_positions, step_positions


def collect_outward(W, start, D=1.0, n_traj=200, T=600, seed=0, drift=0.25, min_speed=0.25):
    """Outward-biased persistent random walks, every one beginning at the centre of cell
    `start`. `drift` is the per-step heading jitter in radians; speed is uniform on
    [min_speed, 1] of full speed."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n_traj, T + 1, 2), np.float32)
    A = np.zeros((n_traj, T, 2), np.float32)
    P[:, 0] = cell_centre(W, start)
    home_pos = P[:, 0].copy()
    heading = rng.uniform(0, 2 * np.pi, n_traj)
    outward = 0.0 if D <= 1 else 1.0 - 1.0 / D

    for t in range(T):
        here = P[:, t]
        # along the heading, at a speed of `min_speed`..1 of full (full = the larger component
        # saturates the action box)
        act = np.stack([np.cos(heading), np.sin(heading)], -1)
        act = act / np.abs(act).max(1, keepdims=True) * rng.uniform(min_speed, 1, (n_traj, 1))
        A[:, t] = act
        P[:, t + 1] = step_positions(W, here, act)
        # a wall (the move came up short) or a random turn ends the run. A voluntary turn
        # refuses the way home with probability `outward`; a forced one is unconstrained.
        blocked = np.linalg.norm(P[:, t + 1] - here, axis=-1) < 0.9 * W["step"] * np.abs(act).max(1)
        voluntary = rng.random(n_traj) < 1.0 / D
        home = descent_direction(W, P[:, t + 1], home_pos)
        away = np.arctan2(home[:, 1], home[:, 0]) + np.pi + rng.uniform(-np.pi / 2, np.pi / 2, n_traj)
        uniform = rng.uniform(0, 2 * np.pi, n_traj)
        fresh = np.where(voluntary & ~blocked & (rng.random(n_traj) < outward), away, uniform)
        heading = np.where(blocked | voluntary, fresh, heading + rng.normal(0, drift, n_traj))
    return P, A


def uniform_batches(W, h, n_batches, batch=256, seed=1):
    """The ceiling: state, action chunk and goal all uniform over free space."""
    rng = np.random.default_rng(seed)
    gamma = W["gamma"]
    shape = (n_batches, batch)
    state = sample_positions(W, n_batches * batch, rng).reshape(shape + (2,))
    goal = sample_positions(W, n_batches * batch, rng).reshape(shape + (2,))
    seq = rng.uniform(-1, 1, shape + (2 * h,)).astype(np.float32)

    reward = np.zeros(shape, np.float32)
    discount = np.full(shape, gamma**h, np.float32)
    cur, done = state.copy(), np.zeros(shape, bool)
    for k in range(h):
        cur = step_positions(W, cur, seq[..., 2 * k:2 * k + 2])
        hit = (~done) & (np.linalg.norm(cur - goal, axis=-1) < W["goal_radius"])
        reward[hit] = gamma**k
        discount[hit] = 0.0
        done |= hit
    return dict(state=state, goal=goal, seq=seq, reward=reward, discount=discount, next_state=cur)


# The same suite as the gridworld study; see gridworld/coverage.py for the reasoning behind the
# InfoNCE directions. sa_g's InfoNCE normalizes over (s, a) -- the axis a greedy policy compares
# along -- and sg_a's over action chunks, which for it is the same thing.
CRITICS = [
    ("monolithic_td", "monolithic", "td"),
    ("sa_g_td", "sa_g", "td"),
    ("sg_a_td", "sg_a", "td"),
    ("sa_g_infonce", "sa_g", "bwd_infonce"),
    ("sg_a_infonce", "sg_a", "infonce"),
    ("sa_g_binary_nce", "sa_g", "binary_nce"),
    ("sg_a_binary_nce", "sg_a", "binary_nce"),
    ("monolithic_mse", "monolithic", "mse"),
    ("sa_g_mse", "sa_g", "mse"),
    ("sg_a_mse", "sg_a", "mse"),
]


def train_suite(W, h, batches, td_batches=None, cand=None, seed=0, kind="td3", lr=3e-4,
                width=256, depth=2, repr_dim=64, lse=0.1, n_freq=16):
    """Train every critic in CRITICS on one dataset.

    `batches` (geometric hindsight goals) trains the contrastive critics; `td_batches` (uniform
    hindsight goals) trains TD and the regression control, defaulting to `batches`. `cand` is
    the candidate pool the sampled max runs over. `width`, `depth` and `repr_dim` size every
    network; `lse` is the contrastive logsumexp penalty.
    """
    import jax

    from .critics import monolithic, sa_g_bilinear, sg_a_bilinear
    from .data import add_optimal_targets
    from .training import candidates, train_contrastive, train_regression, train_td

    if cand is None:
        cand = candidates(h, 64)
    td_batches = add_optimal_targets(W, batches if td_batches is None else td_batches)
    scale = 1.0 / W["n"]

    out = {}
    for i, (name, factor, objective) in enumerate(CRITICS):
        key = jax.random.PRNGKey(seed + i)
        energy = "dot" if objective in ("td", "mse") else "norm"
        build = {"monolithic": lambda: monolithic(h, scale, width, depth, n_freq),
                 "sa_g": lambda: sa_g_bilinear(h, scale, repr_dim, energy, width, depth, n_freq),
                 "sg_a": lambda: sg_a_bilinear(h, scale, repr_dim, energy, width, depth, n_freq)}[factor]()
        if objective == "td":
            p, q, losses, accs = train_td(build, td_batches, cand, kind, key, lr=lr)
        elif objective == "mse":
            p, q, losses, accs = train_regression(build, td_batches, key, lr=lr)
        else:
            p, q, losses, accs = train_contrastive(build, batches, key, lr=lr, loss_fn=objective,
                                                   logsumexp_coeff=lse)
        out[name] = dict(params=p, q=q, cand=cand, losses=losses, accs=accs, objective=objective,
                         n_params=sum(x.size for x in jax.tree_util.tree_leaves(p)))
        yield name, out
