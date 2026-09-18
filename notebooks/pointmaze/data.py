"""Random-action rollouts, relabeled the way the agents relabel them."""

import numpy as np

from .world import GAMMA, sample_positions, step_positions


def collect(W, n_traj=300, T=None, seed=0):
    """Uniform random actions. As in the discrete world, nothing here is goal-directed."""
    if T is None:
        T = int(np.clip(6 * W["n"] / W["step"], 100, 1500))
    rng = np.random.default_rng(seed)
    pos = sample_positions(W, n_traj, rng)
    P = np.zeros((n_traj, T + 1, 2), np.float32)
    A = rng.uniform(-1, 1, (n_traj, T, 2)).astype(np.float32)
    P[:, 0] = pos
    for t in range(T):
        pos = step_positions(W, pos, A[:, t])
        P[:, t + 1] = pos
    return P, A


def potential(W, pos, goal, beta):
    """Phi = -beta * straight-line distance, normalized. Euclidean on purpose: it is what an agent
    could compute, and in a maze it is often wrong."""
    if beta == 0:
        return np.zeros(pos.shape[:-1], np.float32)
    d = np.linalg.norm(pos - goal, axis=-1)
    return (-beta * d / (W["n"] * np.sqrt(2))).astype(np.float32)


def make_batches(W, P, A, h, n_batches, batch=256, seed=1, shaping=0.0):
    """Chunked transitions with a geometric future goal, the m-step return, and where it landed."""
    rng = np.random.default_rng(seed)
    n_traj, T = A.shape[0], A.shape[1]
    traj = rng.integers(0, n_traj, (n_batches, batch))
    t = rng.integers(0, T - h, (n_batches, batch))
    j = np.minimum(t + rng.geometric(1 - GAMMA, (n_batches, batch)), T)

    goal = P[traj, j]                                   # a future position on the same trajectory
    state = P[traj, t]
    seq = np.stack([A[traj, t + k] for k in range(h)], -2).reshape(n_batches, batch, h * 2)

    reward = np.zeros(state.shape[:-1], np.float32)
    discount = np.full(state.shape[:-1], GAMMA**h, np.float32)
    done = np.zeros(state.shape[:-1], bool)
    for k in range(h):
        hit = (~done) & (np.linalg.norm(P[traj, t + k + 1] - goal, axis=-1) < W["goal_radius"])
        reward[hit] = GAMMA**k                          # the goal absorbs
        discount[hit] = 0.0
        done |= hit
    nxt = P[traj, t + h]

    if shaping:
        reward = reward + discount * potential(W, nxt, goal, shaping) \
            - potential(W, state, goal, shaping)
    return dict(state=state, goal=goal, seq=seq, reward=reward, discount=discount, next_state=nxt)
