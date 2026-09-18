"""CARL vs. its baselines on a continuous point maze.

The discrete sibling is `gridworld_carl.py`; the machinery is in `notebooks/pointmaze/`. Two
things change once the action space is continuous: the exact max over chunks becomes a sampled
max over a candidate pool, and "optimal" becomes the descent direction of a Dijkstra geodesic
field rather than one of four moves.

Run it with:  uv run --with marimo marimo edit notebooks/pointmaze_carl.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    for _p in ("notebooks", "."):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pointmaze as P

    return P, mo, np, plt


@app.cell
def _(mo):
    size = mo.ui.slider(9, 15, value=11, step=2, label="maze", show_value=True)
    seed = mo.ui.slider(0, 9, value=0, label="maze seed", show_value=True)
    chunk = mo.ui.slider(1, 4, value=3, label="chunk h", show_value=True)
    shaping = mo.ui.slider(0.0, 2.0, value=0.0, step=0.5, label="shaping beta", show_value=True)
    n_cand = mo.ui.slider(16, 256, value=64, step=16, label="candidates", show_value=True)
    steps = mo.ui.slider(400, 3000, value=1200, step=200, label="steps", show_value=True)
    train = mo.ui.run_button(label="train all eight critics")

    mo.hstack([size, seed, chunk, shaping, n_cand, steps, train],
              justify="start", gap=1, wrap=True)
    return chunk, n_cand, seed, shaping, size, steps, train


@app.cell
def _(P, chunk, mo, n_cand, seed, shaping, size, steps, train):
    W = P.build_world(size.value, seed=seed.value)
    trained, results, floor = {}, {}, 0.0

    if train.value:
        with mo.status.progress_bar(total=8, title="training") as _bar:
            for _name, _out in P.train_all(W, h=chunk.value, steps=steps.value,
                                           shaping=shaping.value, n_cand=n_cand.value):
                trained = _out
                _bar.update()
        results = {_k: P.evaluate(W, _v, n_eval=150, n_probe=250) for _k, _v in trained.items()}
        floor = P.random_alignment(W)

    mo.md(
        f"Point maze {W['n']}x{W['n']}, {len(W['free_cells'])} open cells, step {W['step']}, "
        f"goal radius {W['goal_radius']}. "
        + (f"Trained **{len(trained)}** critics, {steps.value} steps each, over a pool of "
           f"{n_cand.value} candidate chunks. A uniformly random action scores "
           f"**{floor:+.3f}** on alignment, which is the floor the bars sit on."
           if trained else "Press **train all eight critics**.")
    )
    return W, floor, results, trained


@app.cell
def _(mo):
    goal_pick = mo.ui.slider(0, 19, value=0, label="goal", show_value=True)
    goal_pick
    return (goal_pick,)


@app.cell
def _(floor, mo, plt, results, trained):
    mo.stop(not trained, mo.md("*(train to see scores)*"))

    def _figure():
        names = list(trained)
        cols = ["#2a7d4f" if k.startswith("CARL") else "#b5651d" if k.startswith("sa_g")
                else "#777" for k in names]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        for ax, key, title in [(axes[0], "success", "success rate"),
                               (axes[1], "align", "alignment with the geodesic descent direction")]:
            vals = [results[k][key] for k in names]
            ax.bar(range(len(names)), vals, color=cols)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(0, color="k", lw=0.8)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7)
        axes[0].set_ylim(0, 1)
        axes[1].axhline(floor, color="crimson", ls="--", lw=1, label=f"random ({floor:+.2f})")
        axes[1].set_ylim(-1, 1); axes[1].legend(fontsize=7)
        fig.suptitle("grey = monolithic   orange = f([s;a])·g(goal)   green = phi([s;g])·psi(a_tau)",
                     fontsize=9)
        return fig

    _figure()
    return


@app.cell
def _(P, W, goal_pick, mo, np, plt, trained):
    mo.stop(not trained, mo.md(""))

    def _figure():
        goal = P.sample_positions(W, 20, np.random.default_rng(7))[int(goal_pick.value)]
        panels = [("geodesic distance", -P.geodesic_grid(W, goal))]
        for name, e in trained.items():
            v = P.value_grid(W, e, goal)
            lo, hi = np.nanmin(v), np.nanmax(v)
            panels.append((name, (v - lo) / (hi - lo + 1e-9)))

        fig, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels), 2.5),
                                 constrained_layout=True)
        for ax, (name, img) in zip(axes, panels):
            im = ax.imshow(img, cmap="coolwarm", origin="upper",   # warm = high value
                           extent=[0, W["n"], W["n"], 0])
            ax.plot(goal[1], goal[0], "*", color="gold", ms=11, mec="black", mew=0.5)
            ax.set_title(name, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
        fig.suptitle("value over position, min-max normalized; leftmost is the true geodesic "
                     "(red star = goal)", fontsize=9)
        return fig

    _figure()
    return


@app.cell
def _(P, W, goal_pick, mo, np, plt, trained):
    mo.stop(not trained, mo.md(""))

    def _figure():
        goal = P.sample_positions(W, 20, np.random.default_rng(7))[int(goal_pick.value)]
        ideal_pos, ideal_vec = P.ideal_field(W, goal)
        panels = [("ideal descent", ideal_pos, ideal_vec)]
        for name, e in trained.items():
            pos, vec = P.action_field(W, e, goal)
            panels.append((name, pos, vec))

        fig, axes = plt.subplots(3, 3, figsize=(9.6, 9.8), constrained_layout=True)
        for ax, (name, pos, vec) in zip(axes.ravel(), panels):
            ax.imshow(np.where(W["walls"], 1.0, np.nan), cmap="Greys", vmin=0, vmax=1,
                      extent=[0, W["n"], W["n"], 0])
            if name == "ideal descent":
                col = "#2a7d4f"
            else:
                col = np.where((vec * P.descent_direction(
                    W, P.geodesic_field(W, goal), pos)).sum(-1) > 0.3, "#2a7d4f", "#c0392b")
            ax.quiver(pos[:, 1], pos[:, 0], vec[:, 1], vec[:, 0], color=col,
                      angles="xy", scale_units="xy", scale=2.0, width=0.007)
            ax.plot(goal[1], goal[0], "*", color="gold", ms=13, mec="black", mew=0.5)
            ax.set_title(name, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.suptitle("the action each critic picks — green points along the geodesic, red does not",
                     fontsize=10)
        return fig

    _figure()
    return


@app.cell
def _(W, mo, np, plt, results, trained):
    mo.stop(not trained, mo.md(""))

    def _figure():
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), constrained_layout=True)
        for ax, (name, r) in zip(axes.ravel(), results.items()):
            ax.imshow(np.where(W["walls"], 1.0, np.nan), cmap="Greys", vmin=0, vmax=1,
                      extent=[0, W["n"], W["n"], 0])
            v = r["visits"]
            if len(v):
                _, _, _, im = ax.hist2d(v[:, 1], v[:, 0], bins=W["n"] * 2,
                                        range=[[0, W["n"]], [0, W["n"]]], cmap="coolwarm",
                                        alpha=0.9)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                cb.ax.tick_params(labelsize=6); cb.set_label("visits", fontsize=6)
            ax.set_ylim(W["n"], 0)
            ax.set_title(f"{name}\nsuccess {r['success']:.2f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("state visitation during evaluation rollouts", fontsize=9)
        return fig

    _figure()
    return


@app.cell
def _(mo, plt, trained):
    mo.stop(not trained, mo.md(""))

    def _figure():
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
        for name, e in trained.items():
            ls = "-" if name.startswith("CARL") else ":" if name.startswith("sa_g") else "--"
            ax = axes[0] if name.endswith("CRL") else axes[1]
            ax.plot(e["losses"][10:], ls, label=name, lw=1)
        axes[0].set_title("contrastive loss", fontsize=10)
        axes[1].set_title("TD mean squared error", fontsize=10); axes[1].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("gradient step"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        return fig

    _figure()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
