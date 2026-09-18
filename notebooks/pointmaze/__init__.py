"""The gridworld comparison, made continuous: a point mass in a maze.

Same eight critics and the same three factorizations as `../gridworld`, with two changes that
follow from a continuous action space:

* There is no enumerable chunk space, so the exact max and logsumexp become a **sampled** max over
  a fixed pool of candidate chunks drawn from [-1, 1]^(2h) -- the QT-Opt approximation. The setup
  stays actor-free; the target is a lower bound on the true max, biased down when the pool is small.
* "Optimal" is no longer a discrete move. Ground truth is the geodesic distance around the walls
  between nodes of a fine grid (all pairs, Dijkstra once), and a critic is scored by the angle
  between the action it picks and the direction that most reduces it. Random actions land within
  45 degrees a quarter of the time, so the floor is visible.

Three topologies: a perfect maze (a tree), a braided maze (the same maze with dead ends opened
into loops) and a grid of rooms. `coverage.py` is the continuous twin of the gridworld's: the
outward-biased collector, the uniform control dataset and the critic suite.

Motion is a point mass with wall sliding: a blocked diagonal still moves along whichever axis is
open, which is what makes the value function interesting near corners.
"""

from .critics import energy_fn, monolithic, sa_g_bilinear, sg_a_bilinear
from .data import add_optimal_targets, collect, make_batches, optimal_q, potential
from .evaluate import (
    action_field,
    cell_grid,
    evaluate,
    geodesic_grid,
    greedy_action,
    ideal_field,
    q_over_candidates,
    random_floor,
    rollouts,
    value_grid,
)
from .training import candidates, train_all, train_contrastive, train_regression, train_td
from .world import (
    build_world,
    cell_centre,
    cell_of,
    descent_direction,
    geodesic,
    is_free,
    sample_positions,
    step_positions,
)

__all__ = [
    "build_world", "is_free", "step_positions", "sample_positions", "cell_of",
    "cell_centre", "geodesic", "descent_direction",
    "collect", "make_batches", "potential", "optimal_q", "add_optimal_targets",
    "energy_fn", "monolithic", "sa_g_bilinear", "sg_a_bilinear",
    "candidates", "train_all", "train_contrastive", "train_regression", "train_td",
    "q_over_candidates", "greedy_action", "rollouts", "evaluate", "random_floor",
    "cell_grid", "action_field", "ideal_field", "value_grid", "geodesic_grid",
]
