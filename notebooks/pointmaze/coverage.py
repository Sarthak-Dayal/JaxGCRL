"""Data distributions of controllable coverage in the continuous maze, from random to full.

The continuous twin of `gridworld/coverage.py`. The dial is the same outward-biased policy: with
probability `1 - 1/D` take the action that most increases the geodesic distance from a moving
anchor, otherwise act randomly and re-anchor. The candidates for an outward move are the eight
compass directions at full speed. Distance has to be the fine geodesic rather than the coarse
cell BFS: a half-cell step that stays inside its cell makes no progress on the cell graph, so
steering by cells reads every other step as stuck and re-anchors, and D stops mattering past 4.
D = 1 is a uniformly random walk in [-1, 1]^2; `uniform` draws state, action and goal
independently over free space.
"""

import numpy as np

from .world import cell_centre, geodesic, sample_positions, step_positions

COMPASS = np.array([(np.cos(t), np.sin(t)) for t in np.arange(8) * np.pi / 4], np.float32)
COMPASS /= np.abs(COMPASS).max(1, keepdims=True)              # full speed along each axis


def collect_outward(W, start, D=1.0, n_traj=200, T=600, seed=0):
    """Random walks biased outward from a moving anchor, every one beginning at the centre of
    cell `start`. See `gridworld.coverage.collect_outward` for why the anchor moves."""
    gamma = 0.0 if D <= 1 else 1.0 - 1.0 / D
    rng = np.random.default_rng(seed)
    P = np.zeros((n_traj, T + 1, 2), np.float32)
    A = np.zeros((n_traj, T, 2), np.float32)
    P[:, 0] = cell_centre(W, start)
    anchor = np.repeat(P[:, 0][:, None, :], 8, 1)                # (n_traj, 8, 2), for geodesic

    for t in range(T):
        here = P[:, t]
        # where each compass move would land, and how far from the anchor that is
        land = np.stack([step_positions(W, here, np.repeat(c[None], n_traj, 0)) for c in COMPASS], 1)
        gain = geodesic(W, anchor, land) + rng.uniform(0, 1e-3, (n_traj, 8))
        go_out = rng.random(n_traj) < gamma
        A[:, t] = np.where(go_out[:, None], COMPASS[np.argmax(gain, 1)],
                           rng.uniform(-1, 1, (n_traj, 2)))
        # A random step restarts the excursion; so does running out of room (a dead end, or the
        # far wall), else a long excursion ping-pongs there.
        stuck = gain.max(1) <= geodesic(W, anchor[:, 0], here) + 1e-3
        keep = go_out & ~stuck
        anchor = np.where(keep[:, None, None], anchor, np.repeat(here[:, None, :], 8, 1))
        P[:, t + 1] = step_positions(W, here, A[:, t])
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
