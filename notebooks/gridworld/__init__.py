"""A gridworld where CARL's critic factorization can be scored against the right answer.

Three ways to factor a goal-conditioned critic, crossed with the objectives that fit them:

    factorization                        contrastive   soft TD    clipped double-Q
    ------------------------------------------------------------------------------
    Q([s; g; a])            monolithic        -          SAC            TD3
    energy(f([s; a]), g(g))    CRL's          CRL      sa_g_sac       sa_g_td3
    energy(phi([s; g]), psi(a_tau))  CARL's   CARL-CRL  CARL-SAC       CARL-TD3

`SAC` vs `sa_g_sac` isolates the factorization with everything else held fixed; `sa_g_sac` vs
`CARL-SAC` isolates which side the goal sits on; `CARL-SAC` additionally chunks.

A gridworld buys three things a brax env cannot. The chunk space is finite -- 4^n sequences -- so
the max and logsumexp over next action sequences are exact and no actor network is needed
anywhere; "optimize the Q function" is literally an argmax. BFS gives the optimal Q, so accuracy
is measurable rather than inferred from returns. And all eight critics train on CPU in under a
minute.

What each objective implements, and where it departs from the paper it is named after:

* `contrastive` -- InfoNCE with the batch as negatives, as in Contrastive RL (Eysenbach et al.,
  2022) and this repo's `crl/losses.py`. The negatives axis follows the factorization: CRL scores
  each (s, a) against every goal in the batch, CARL scores each (s, g) against every action
  sequence. Note this critic estimates a discounted state-occupancy ratio rather than Q itself, so
  it is scored by rank agreement with the optimal Q, not by absolute error.

* `sac` -- the soft TD backup, with the next-state value taken exactly as
  `alpha * logsumexp(Q(s', .) / alpha)` over all chunks. With finitely many actions this is the
  exact soft value, so no actor and no entropy tuning are needed.

* `td3` -- clipped double-Q: two heads, `min` over them in the target, greedy over chunks.
  Target-policy smoothing has no discrete analogue and is omitted, as are delayed policy updates
  (there is no policy to delay).

The chunked variants take a standard TD backup over `Q(s_t, a_{t:t+n})`, which is an unbiased
n-step backup precisely because the whole sequence is an input -- the Q-chunking result
(https://arxiv.org/abs/2507.07969). Unlike Q-chunking, the greedy policy here replans every step
and executes only the chunk's first action, matching `action_exec_len = 1` in the repo's agents.

`../gridworld_carl.py` is the marimo front end.
"""

from .critics import energy_fn, monolithic, sa_g_bilinear, sg_a_bilinear
from .data import add_optimal_targets, collect, make_batches, optimal_q
from .evaluate import (
    agree_by_distance,
    data_coverage,
    evaluate,
    greedy_action,
    greedy_field,
    optimal_field,
    q_over_chunks,
    rollouts_to_goal,
    to_grid,
    value_map,
)
from .training import train_all, train_contrastive, train_regression, train_td
from .world import ACTIONS, GAMMA, N_ACT, all_chunks, build_world

__all__ = [
    "ACTIONS", "GAMMA", "N_ACT", "all_chunks", "build_world",
    "add_optimal_targets", "collect", "make_batches", "optimal_q",
    "energy_fn", "monolithic", "sa_g_bilinear", "sg_a_bilinear",
    "train_all", "train_contrastive", "train_regression", "train_td",
    "agree_by_distance", "data_coverage", "evaluate", "greedy_action", "greedy_field", "optimal_field", "q_over_chunks",
    "rollouts_to_goal", "to_grid", "value_map",
]
