"""Random-walk data, relabeled the way the agents relabel it, plus the optimal Q."""

import numpy as np

from .world import GAMMA, N_ACT


def collect(W, n_traj=400, T=None, seed=0):
    """Uniform random walks. Nothing here is goal-directed -- the critics only ever see relabeled
    data, which is the point of the comparison.

    Relabeled goals can only come from within a trajectory, so a walk that never travels as far as
    the goals it will be evaluated on leaves the critic unable to rank them -- in a maze that
    dominates every other effect. Random walks are diffusive: covering graph distance d takes on
    the order of d^2 steps, not d. Hence T ~ mean distance squared. Even so a maze much past 13
    cells across is out of reach of uniform random walks, and every critic sits near chance there.
    """
    if T is None:
        _d = W["dist"][np.isfinite(W["dist"])].mean()
        T = int(np.clip(2 * _d**2, 40, 2000))
    rng = np.random.default_rng(seed)
    S = np.zeros((n_traj, T + 1), np.int32)
    A = rng.integers(0, N_ACT, (n_traj, T)).astype(np.int32)
    S[:, 0] = rng.integers(0, W["N"], n_traj)
    for t in range(T):
        S[:, t + 1] = W["nxt"][S[:, t], A[:, t]]
    return S, A


def potential(W, states, goals, beta):
    """Phi(s) = -beta * straight-line distance to the goal, normalized by the grid diagonal.

    Deliberately Euclidean rather than BFS: it is the signal an agent could actually compute, and
    in a maze it is often wrong -- the way out of a dead end leads away from the goal. That is what
    makes the shaped Q interesting rather than a hint.
    """
    if beta == 0:
        return np.zeros(np.shape(states), np.float32)
    d = np.linalg.norm(W["cells"][states] - W["cells"][goals], axis=-1)
    return (-beta * d / np.hypot(W["n"], W["n"])).astype(np.float32)


def make_batches(W, S, A, n, n_batches, batch=256, seed=1, shaping=0.0, future="geometric"):
    """Relabeled transitions: a hindsight goal from the trajectory's future, the chunk executed
    from t, the m-step return under an absorbing goal, and where the chunk landed.

    `future` picks the hindsight goal. "geometric" draws it gamma-discounted steps ahead -- the
    discounted state occupancy, which is what a contrastive critic's positives have to be for
    its estimate to mean anything. "uniform" draws it uniformly from the rest of the trajectory,
    HER's `future` strategy; a TD critic makes no assumption about where the goal came from, so
    it takes the version with the widest reach.

    t is drawn so a chunk never runs off the end of its trajectory, which is the gridworld's
    stand-in for the clipping `fold_action_seq` does with `within`/`num_actions`.
    """
    rng = np.random.default_rng(seed)
    n_traj, T = A.shape
    traj = rng.integers(0, n_traj, (n_batches, batch))
    t = rng.integers(0, T - n, (n_batches, batch))

    gam = W["gamma"]
    if future == "geometric":
        j = np.minimum(t + rng.geometric(1 - gam, (n_batches, batch)), T)
    else:
        j = rng.integers(t + 1, T + 1)
    goal = S[traj, j]
    state = S[traj, t]
    seq = np.stack([A[traj, t + k] for k in range(n)], -1)

    reward = np.zeros(state.shape, np.float32)
    discount = np.full(state.shape, gam**n, np.float32)
    cur, done = state.copy(), np.zeros(state.shape, bool)
    for k in range(n):
        cur = W["nxt"][cur, seq[..., k]]
        hit = (~done) & (cur == goal)
        reward[hit] = gam**k                                            # the goal absorbs
        discount[hit] = 0.0
        done |= hit
    # Potential-based shaping (Ng et al., 1999): adding gamma*Phi(s') - Phi(s) to the reward
    # leaves the optimal policy unchanged, and over a chunk the intermediate terms telescope to
    # gamma^m * Phi(s_{t+m}) - Phi(s_t). The optimal Q shifts by exactly -Phi(s), a constant in the
    # action, so BFS remains the ground truth for which move is best.
    if shaping:
        reward = reward + discount * potential(W, cur, goal, shaping) \
            - potential(W, state, goal, shaping)

    return dict(state=state, goal=goal, seq=seq, reward=reward, discount=discount, next_state=cur)


def optimal_q(W, states, goals, chunks, shaping=0.0):
    """gamma^k if the chunk first reaches the goal on step k+1, else gamma^n * V*(landing)."""
    n = chunks.shape[1]
    gam = W["gamma"]
    s, q = states.copy(), np.zeros(len(states), np.float32)
    done = np.zeros(len(states), bool)
    for k in range(n):
        s = W["nxt"][s, chunks[:, k]]
        hit = (~done) & (s == goals)
        q[hit] = gam**k
        done |= hit
    tail = gam**n * gam ** W["dist"][goals, s]
    q[~done] = tail[~done]
    return q - potential(W, states, goals, shaping)      # the shift shaping induces


def add_optimal_targets(W, batches, shaping=0.0):
    """Attach the exact optimal Q of every (state, chunk, goal) in a batch.

    Regressing on this is the control the RL objectives are missing: same data, same architecture,
    perfect targets. Whatever it cannot fit is a limit of the coverage or the network, not of the
    objective being tested.
    """
    shape = batches["state"].shape
    q = optimal_q(W, batches["state"].ravel(), batches["goal"].ravel(),
                  batches["seq"].reshape(-1, batches["seq"].shape[-1]), shaping)
    return {**batches, "q_star": q.reshape(shape).astype(np.float32)}
