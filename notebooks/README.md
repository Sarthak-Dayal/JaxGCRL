# Notebooks

Interactive [marimo](https://marimo.io) notebooks for studying goal-conditioned critics in worlds
where the optimal Q is known exactly. Each notebook is a plain Python file; the `gridworld/` and
`pointmaze/` packages next to them hold the environments, critics and training code, and are
self-contained (they do not import from `jaxgcrl`).

| notebook | what it studies |
|---|---|
| `coverage_study.py` | How much of a critic's accuracy is the objective, and how much is the data it was shown. Three factorizations (monolithic, CRL's `sa_g`, CARL's `sg_a`) under TD, contrastive (InfoNCE, binary NCE) and oracle-regression objectives, trained on datasets of controllable coverage in a maze, then scaled over network size. |
| `gridworld_carl.py` | CARL against its baselines on a gridworld, including action chunking. |
| `pointmaze_carl.py` | The same comparison on a continuous point maze. |
| `q_landscape.py`, `render_runs.py` | Inspecting trained `jaxgcrl` runs. |

## Running `coverage_study.py`

Everything runs inside the repo's nix FHS environment (that is where JAX finds CUDA). Build it once:

```
nix build .#fhs-dev
```

Then, from the repo root, start marimo. `--with` pulls marimo and plotly into the run without
touching the project's lockfile; `JAXGCRL_SKIP_SYNC=1` skips the `uv sync` that the FHS entrypoint
otherwise performs on every launch.

```
JAXGCRL_SKIP_SYNC=1 ./result/bin/fhs-ubuntu-dev \
  uv run --with marimo --with plotly \
  marimo edit --headless --host 127.0.0.1 --port 2718 notebooks/coverage_study.py
```

marimo prints a URL containing an access token. If the machine is remote, forward the port from
your laptop and open that URL with the host replaced by `localhost`:

```
ssh -N -L 2718:127.0.0.1:2718 <user>@<host>
```

### What you will see

The notebook is a sequence of sections, each with its controls directly above the figure they
change. Everything before the training button re-runs instantly when a slider moves.

1. **Maze** -- size and seed. Hover a cell for its index; the start/goal sliders take indices.
2. **Datasets** -- a start cell, the number of walks and their length. Every walk starts at the
   same cell and is biased outward with effective excursion length `D` (`D=1` is a random walk);
   `uniform` draws state, action and goal independently. The figure shows visits per cell.
3. **Training** -- gradient steps and learning rate, then a button that trains all ten critics on
   all five datasets on the GPU (a few minutes). Training curves appear below it.
4. **Critic grid** -- for a chosen dataset and goal: a row per objective, a column per
   factorization, the oracle at the left. Colour is `V(s, g)`; the arrow at each cell is the greedy
   move, black where it is BFS-optimal and magenta where it is not; hover for every action's Q.
5. **Accuracy against D** -- the share of cells where each critic picks a BFS-optimal move, per
   dataset; click the legend to hide or isolate a critic.
6. **Parameter scaling** -- the same critics at five network widths on one dataset: accuracy
   against parameter count per critic, and the same grid view with a width selector.

Training runs on the GPU; evaluation runs on the CPU (it is hundreds of tiny calls, which the GPU
does slowly). Both are single compiled `lax.scan`s, so a full sweep is dominated by XLA compile time.

The docstring at the top of `coverage_study.py` states the question the notebook answers and how the
datasets are constructed; `gridworld/coverage.py` documents the critics, including why `sa_g`'s
InfoNCE is taken in the backward direction.
