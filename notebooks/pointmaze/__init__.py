"""The gridworld comparison, made continuous: a point mass in a maze.

Same eight critics and the same three factorizations as `../gridworld`, with two changes that
follow from a continuous action space:

* There is no enumerable chunk space, so the exact max and logsumexp become a **sampled** max over
  a fixed pool of candidate chunks drawn from [-1, 1]^(2h) -- the QT-Opt approximation. The setup
  stays actor-free; the target is a lower bound on the true max, biased down when the pool is small.
* "Optimal" is no longer a discrete move. Ground truth is a Dijkstra geodesic field on a fine grid
  around the walls, and a critic is scored by the cosine between the action it picks and the
  direction that most reduces that field. Random actions score ~0 on it, so the floor is visible.

Motion is a point mass with wall sliding: a blocked diagonal still moves along whichever axis is
open, which is what makes the value function interesting near corners.
"""

from .critics import energy_fn, monolithic, sa_g_bilinear, sg_a_bilinear
from .data import collect, make_batches, potential
from .evaluate import (
    action_field,
    best_chunk,
    evaluate,
    geodesic_grid,
    greedy_action,
    ideal_field,
    random_alignment,
    value_grid,
)
from .training import candidates, train_all, train_contrastive, train_td
from .world import (
    GAMMA,
    build_world,
    descent_direction,
    field_at,
    geodesic_field,
    is_free,
    sample_positions,
    step_positions,
)

__all__ = [
    "GAMMA", "build_world", "is_free", "step_positions", "sample_positions",
    "geodesic_field", "field_at", "descent_direction",
    "collect", "make_batches", "potential",
    "energy_fn", "monolithic", "sa_g_bilinear", "sg_a_bilinear",
    "candidates", "train_all", "train_contrastive", "train_td",
    "best_chunk", "greedy_action", "evaluate", "random_alignment",
    "action_field", "ideal_field", "value_grid", "geodesic_grid",
]
