"""Random-action rollouts, relabeled the way the agents relabel them, plus the optimal Q."""

import numpy as np

from .world import geodesic, sample_positions, step_positions


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


def make_batches(W, P, A, h, n_batches, batch=256, seed=1, shaping=0.0, future="geometric"):
    """Chunked transitions with a hindsight goal from the trajectory's future, the m-step return
    under an absorbing goal, and where the chunk landed.

    `future` picks the goal: "geometric" draws it gamma-discounted steps ahead (the discounted
    occupancy a contrastive critic's positives must come from), "uniform" anywhere in the rest
    of the trajectory (HER's `future` strategy, for TD and the regression control).
    """
    rng = np.random.default_rng(seed)
    gamma = W["gamma"]
    n_traj, T = A.shape[0], A.shape[1]
    traj = rng.integers(0, n_traj, (n_batches, batch))
    t = rng.integers(0, T - h, (n_batches, batch))
    if future == "geometric":
        j = np.minimum(t + rng.geometric(1 - gamma, (n_batches, batch)), T)
    else:
        j = rng.integers(t + 1, T + 1)

    goal = P[traj, j]                                   # a future position on the same trajectory
    state = P[traj, t]
    seq = np.stack([A[traj, t + k] for k in range(h)], -2).reshape(n_batches, batch, h * 2)

    reward = np.zeros(state.shape[:-1], np.float32)
    discount = np.full(state.shape[:-1], gamma**h, np.float32)
    done = np.zeros(state.shape[:-1], bool)
    for k in range(h):
        hit = (~done) & (np.linalg.norm(P[traj, t + k + 1] - goal, axis=-1) < W["goal_radius"])
        reward[hit] = gamma**k                          # the goal absorbs
        discount[hit] = 0.0
        done |= hit
    nxt = P[traj, t + h]

    if shaping:
        reward = reward + discount * potential(W, nxt, goal, shaping) \
            - potential(W, state, goal, shaping)
    return dict(state=state, goal=goal, seq=seq, reward=reward, discount=discount, next_state=nxt)


def optimal_q(W, state, goal, seq):
    """gamma^k if the chunk enters the goal radius on step k+1, else gamma^h * V*(landing).

    V*(x, g) = gamma ** (geodesic distance / step): the return of moving along the shortest path
    around the walls at full speed. The step counts along a geodesic are approximate (a diagonal
    move covers sqrt(2) more than an axis move), but the ordering over actions -- which is all
    the argmax reads -- is the ordering of the geodesic itself.
    """
    h, gamma = seq.shape[-1] // 2, W["gamma"]
    pos, q = state.copy(), np.zeros(len(state), np.float32)
    done = np.zeros(len(state), bool)
    for k in range(h):
        pos = step_positions(W, pos, seq[:, 2 * k:2 * k + 2])
        hit = (~done) & (np.linalg.norm(pos - goal, axis=-1) < W["goal_radius"])
        q[hit] = gamma**k
        done |= hit
    d = geodesic(W, pos, goal)
    tail = gamma**h * gamma ** (np.where(np.isfinite(d), d, 1e3) / W["step"])
    q[~done] = tail[~done]
    return q


def add_optimal_targets(W, batches):
    """Attach the exact optimal Q of every (state, chunk, goal) in a batch: the regression control."""
    shape = batches["state"].shape[:-1]
    q = optimal_q(W, batches["state"].reshape(-1, 2), batches["goal"].reshape(-1, 2),
                  batches["seq"].reshape(-1, batches["seq"].shape[-1]))
    return {**batches, "q_star": q.reshape(shape).astype(np.float32)}
