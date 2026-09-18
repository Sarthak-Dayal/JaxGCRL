"""CARL vs. its baselines on a gridworld where the optimal Q is known.

All the machinery lives in `notebooks/gridworld/`; this file is controls and plots. See that
package's docstring for what each critic implements and where it departs from its namesake.

Run it with:  uv run --with marimo marimo edit notebooks/gridworld_carl.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys

    os.environ.setdefault("JAX_PLATFORMS", "cpu")     # small; keep off a busy GPU
    for _p in ("notebooks", "."):                     # run from the repo root or from notebooks/
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import gridworld as G

    return G, mo, np, plt


@app.cell
def _(mo):
    # Every control in one block, so nothing needs scrolling to reach.
    layout = mo.ui.dropdown(options=["four rooms", "maze", "two rooms", "empty"],
                            value="four rooms", label="topology")
    size = mo.ui.slider(9, 17, value=11, step=2, label="grid", show_value=True)
    shaping = mo.ui.slider(0.0, 2.0, value=0.0, step=0.5, label="shaping beta", show_value=True)
    chunk = mo.ui.slider(1, 4, value=3, label="chunk n", show_value=True)
    steps = mo.ui.slider(500, 4000, value=1500, step=500, label="steps", show_value=True)
    train = mo.ui.run_button(label="train all eight critics")

    mo.hstack([layout, size, chunk, shaping, steps, train], justify="start", gap=1, wrap=True)
    return chunk, layout, shaping, size, steps, train


@app.cell
def _(G, chunk, layout, mo, shaping, size, steps, train):
    # Always defines W/trained/results, so the plot cells below never see an undefined name.
    W = G.build_world(layout.value, size.value)
    trained, results = {}, {}

    if train.value:
        with mo.status.progress_bar(total=8, title="training") as _bar:
            for _name, _out in G.train_all(W, n=chunk.value, steps=steps.value,
                                           shaping=shaping.value):
                trained = _out
                _bar.update()
        results = {_k: G.evaluate(W, _v, shaping=shaping.value) for _k, _v in trained.items()}
        _cov = G.data_coverage(W, next(iter(trained.values()))["data"][0])

    mo.md(
        f"{W['N']}-cell **{layout.value}** world, chunk n={chunk.value}. "
        + (f"Trained **{len(trained)}** critics for {steps.value} steps each.\n\n"
           f"Data reach: the random walks travel a median of **{_cov['walk_median']:.0f}** steps "
           f"(90th pct {_cov['walk_p90']:.0f}) while the evaluation goals sit a mean of "
           f"**{_cov['goal_mean']:.0f}** steps away. When the first number is well below the "
           f"second the critics are being asked to rank goals no trajectory ever reached, and the "
           f"bars below collapse toward the **{_cov['chance']:.2f}** a random action scores."
           if trained else "Press **train all eight critics**.")
    )
    return W, results, trained


@app.cell
def _(W, mo, np):
    # A goal to inspect. Defaults to the cell furthest from the corner, so the field has structure.
    goal_pick = mo.ui.slider(0, W["N"] - 1, value=int(np.argmax(W["dist"][:, 0])),
                             label="goal cell", show_value=True)
    goal_pick
    return (goal_pick,)


@app.cell
def _(mo, plt, results, trained):
    mo.stop(not trained, mo.md("*(train to see scores)*"))

    def _figure():
        names = list(trained)

        def bars(ax, key, title, lo=0.0):
            vals = [results[k][key] for k in names]
            cols = ["#2a7d4f" if k.startswith("CARL") else "#b5651d" if k.startswith("sa_g")
                    else "#777" for k in names]
            ax.bar(range(len(names)), vals, color=cols)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.set_ylim(lo, 1)
            ax.grid(axis="y", alpha=0.3)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
        bars(axes[0], "success", "success rate")
        bars(axes[1], "agree", "picks a BFS-optimal move")
        bars(axes[2], "coverage", "state coverage")
        bars(axes[3], "rank", "rank agreement with optimal Q", lo=-1)
        fig.suptitle("grey = monolithic   orange = f([s;a])·g(goal)   green = phi([s;g])·psi(a_tau)",
                     fontsize=9)
        return fig

    _figure()
    return


@app.cell
def _(G, W, goal_pick, mo, plt, trained):
    mo.stop(not trained, mo.md(""))

    def _figure():
        goal = int(goal_pick.value)
        # Each critic sits on its own scale, so each panel carries its own colorbar rather than one
        # shared one -- the shape is comparable, the units are not.
        panels = [("optimal V", G.to_grid(W, G.GAMMA ** W["dist"][goal]))]
        for name, e in trained.items():
            panels.append((name, G.to_grid(W, G.value_map(W, e, goal))))

        fig, axes = plt.subplots(1, len(panels), figsize=(2.5 * len(panels), 2.9),
                                 constrained_layout=True)
        gr, gc = W["cells"][goal]
        for ax, (name, img) in zip(axes, panels):
            im = ax.imshow(img, cmap="coolwarm")          # warm = high value, cold = low
            ax.plot(gc, gr, "*", color="gold", ms=11, mec="black", mew=0.5)
            ax.set_title(name, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
        fig.suptitle("V(s, g) = max over chunks, each on its own scale (gold star = goal)",
                     fontsize=9)
        return fig

    _figure()
    return


@app.cell
def _(G, W, goal_pick, mo, np, plt, trained):
    mo.stop(not trained, mo.md(""))

    def _figure():
        goal = int(goal_pick.value)
        opt = G.optimal_field(W, goal)                   # (N, 4) mask of BFS-optimal actions
        rows, cols = W["cells"][:, 0], W["cells"][:, 1]
        gr, gc = W["cells"][goal]

        def arrows(ax, act, title):
            # walls drawn dark; imshow already inverts y, so dr is used unnegated
            ax.imshow(np.where(W["walls"], 1.0, np.nan), cmap="Greys", vmin=0, vmax=1)
            good = opt[np.arange(W["N"]), act]
            dr, dc = G.ACTIONS[act, 0], G.ACTIONS[act, 1]
            for mask, color in [(good, "#2a7d4f"), (~good, "#c0392b")]:
                if mask.any():
                    ax.quiver(cols[mask], rows[mask], dc[mask], dr[mask], color=color,
                              angles="xy", scale_units="xy", scale=2.2, width=0.011)
            ax.plot(gc, gr, "*", color="gold", ms=14, mec="black", mew=0.5)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        panels = [("BFS optimal", np.argmax(opt, 1))]
        panels += [(k, G.greedy_field(W, v, goal)) for k, v in trained.items()]

        fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.6), constrained_layout=True)
        for ax, (name, act) in zip(axes.ravel(), panels):
            share = opt[np.arange(W["N"]), act].mean()
            arrows(ax, act, name + ("" if name == "BFS optimal" else f"  ({share:.2f} optimal)"))
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.suptitle("argmax action at every cell — green agrees with BFS, red does not", fontsize=10)
        return fig

    _figure()
    return


@app.cell
def _(G, W, goal_pick, mo, np, plt, trained):
    mo.stop(not trained, mo.md(""))

    def _figure():
        goal = int(goal_pick.value)
        gr, gc = W["cells"][goal]
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), constrained_layout=True)
        for ax, (name, e) in zip(axes.ravel(), trained.items()):
            visits, arrived = G.rollouts_to_goal(W, e, goal)
            im = ax.imshow(G.to_grid(W, np.log1p(visits)), cmap="coolwarm")
            ax.plot(gc, gr, "*", color="gold", ms=13, mec="black", mew=0.5)
            ax.set_title(f"{name}\nreached {arrived.mean():.2f} of starts", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6)
            cb.set_label("log(1 + visits)", fontsize=6)
        fig.suptitle("state visitation, log(1 + visits), greedy rollouts from every cell to one "
                     "goal\na single bright cell means the policy oscillates there rather than "
                     "arriving", fontsize=9)
        return fig

    _figure()
    return


@app.cell
def _(mo, plt, trained):
    mo.stop(not trained, mo.md(""))

    def _figure():
        fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
        for name, e in trained.items():
            ls = "-" if name.startswith("CARL") else ":" if name.startswith("sa_g") else "--"
            ax.plot(e["losses"][10:], ls, label=name, lw=1)
        ax.set_yscale("symlog")
        ax.set_xlabel("gradient step"); ax.set_ylabel("critic loss")
        ax.legend(fontsize=7, ncol=4); ax.grid(alpha=0.3)
        return fig

    _figure()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
