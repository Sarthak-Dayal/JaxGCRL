"""How much of a critic's accuracy is the objective, and how much is the data it was shown?

Every critic here is trained on relabeled trajectories, so the (state, goal) pairs it sees are the
pairs some behaviour policy happened to connect. A uniform random walk is diffusive: in a maze it
travels a median of ~1.5 cells before the goal is relabeled, while evaluation asks about goals
tens of cells away. This notebook turns that gap into a dial.

The dial is an outward-biased policy (`gridworld/coverage.py`): with probability `1 - 1/D` it
takes the action that most increases the BFS distance from a moving anchor, otherwise it acts
randomly and re-anchors. D = 1 is the plain random walk; larger D gives longer directed
excursions and therefore more distant pairs. `uniform` -- state, action and goal all drawn
uniformly -- is the ceiling no behaviour policy can reach.

Ten critics: the three factorizations under a TD objective, the two two-tower ones under each
of two contrastive objectives (InfoNCE and binary NCE), and all three regressed directly on the
optimal Q. That last group is the control -- same data, same architecture, perfect targets -- so
anything it cannot reach is a limit of the coverage or the network rather than of the objective.

Each knob sits next to the picture it changes. The maze and the walks are rebuilt the moment a
slider moves -- they are numpy, and cheap. Training all ten critics on every dataset is behind
a button; the D selector below it only reads precomputed results.

Run it with:  uv run --with marimo --with plotly marimo edit notebooks/coverage_study.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    for _p in ("notebooks", "."):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import gridworld as G
    from gridworld import coverage as C
    from gridworld.data import make_batches

    return C, G, go, make_batches, make_subplots, mo, np


@app.cell
def _(G, go, make_subplots, np):
    # Every picture of the maze is a heatmap over the free cells with the walls drawn on top,
    # one panel per thing compared. Hover a cell for its index -- the sliders take indices --
    # and its value.
    ARROW = np.array(["\u2191", "\u2193", "\u2190", "\u2192"])       # up, down, left, right, as ACTIONS

    def grid_figure(n_panels, titles, height=340, cols=None):
        cols = cols or n_panels
        rows = -(-n_panels // cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                            horizontal_spacing=0.1, vertical_spacing=0.12)
        # marimo switches plotly's default template to plotly_dark in its dark theme, which
        # would put white text on these white panels; pin the template so it does not
        fig.update_layout(template="plotly_white", height=height * rows,
                          margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white",
                          paper_bgcolor="white", showlegend=False)
        fig.update_annotations(font_size=11)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, autorange="reversed")           # row 0 at the top
        for i in range(n_panels):                                       # square cells
            fig.update_yaxes(scaleanchor="x" if i == 0 else f"x{i + 1}", scaleratio=1,
                             row=i // cols + 1, col=i % cols + 1)
        return fig

    def heatmap(fig, panel, W, vals, hover, colorscale="Viridis", zmin=None, zmax=None,
                cols=None):
        """One panel: `vals` over the free cells, `hover` a per-cell string, walls in grey, a
        colourbar beside the panel. `panel` is 0-based."""
        cols = cols or len(fig._grid_ref[0])
        r, c = panel // cols + 1, panel % cols + 1
        sub = fig.get_subplot(r, c)
        xd, yd = sub.xaxis.domain, sub.yaxis.domain
        text = np.full(W["walls"].shape, "", dtype=object)
        text[W["cells"][:, 0], W["cells"][:, 1]] = hover
        fig.add_trace(go.Heatmap(
            z=G.to_grid(W, vals), text=text, hoverongaps=False,
            hovertemplate="%{text}<extra></extra>", colorscale=colorscale, zmin=zmin, zmax=zmax,
            colorbar=dict(x=xd[1] + 0.004, y=(yd[0] + yd[1]) / 2, len=0.7 * (yd[1] - yd[0]),
                          thickness=8, tickfont=dict(size=8), xpad=0)),
            row=r, col=c)
        fig.add_trace(go.Heatmap(
            z=np.where(W["walls"], 1.0, np.nan), hoverinfo="skip", showscale=False,
            colorscale=[[0, "#3a3a3a"], [1, "#3a3a3a"]]), row=r, col=c)
        return r, c

    def star(fig, W, cell, r, c):
        gr, gc = W["cells"][cell]
        fig.add_trace(go.Scatter(x=[gc], y=[gr], mode="markers", hoverinfo="skip",
                                 marker=dict(symbol="star", size=13, color="gold",
                                             line=dict(color="black", width=1))), row=r, col=c)

    return ARROW, grid_figure, heatmap, star


@app.cell
def _(mo):
    size = mo.ui.slider(9, 17, value=13, step=2, label="maze size", show_value=True)
    seed = mo.ui.slider(0, 5, value=0, label="maze seed", show_value=True)
    return seed, size


@app.cell
def _(G, grid_figure, heatmap, mo, np, seed, size):
    # A maze, not rooms: long horizon, dead ends, and a value function that has to route around
    # obstacles rather than flow toward the goal. Features are fixed to one-hot: with raw (x, y)
    # the regression control tops out at 0.24 on this maze -- coordinates cannot express a value
    # that jumps across a wall -- so a representation failure would masquerade as a coverage
    # result. One-hot also removes generalization between cells, which is what makes this a
    # clean test of coverage.
    W = G.build_world("maze", size.value, seed=seed.value, features="onehot")

    _fig = grid_figure(1, [""], height=300)
    _fig.update_layout(width=300, margin=dict(l=10, r=10, t=10, b=10))
    heatmap(_fig, 0, W, np.zeros(W["N"]), [f"cell {i} (row {r}, col {c})"
                                           for i, (r, c) in enumerate(W["cells"])],
            colorscale=[[0, "#e8e8e8"], [1, "#e8e8e8"]])
    _fig.data[0].showscale = False

    _d = W["dist"][W["dist"] < 1e9]
    mo.vstack([
        mo.hstack([size, seed], justify="start", gap=1),
        mo.hstack([_fig, mo.md(
            f"**{W['n']}x{W['n']}** maze, **{W['N']}** open cells. Mean pairwise distance "
            f"**{_d.mean():.1f}**, longest **{_d.max():.0f}**. gamma **{W['gamma']:.3f}**, the "
            f"half-life of one mean path, so values stay spread over the distances that exist. "
            f"Hover a cell for its index -- the `start cell` and `goal cell` sliders take "
            f"indices."
        )], justify="start", gap=2, align="center"),
    ])
    return (W,)


@app.cell
def _(W, mo):
    start = mo.ui.slider(0, W["N"] - 1, value=0, label="start cell", show_value=True)
    n_traj = mo.ui.slider(100, 600, value=200, step=100, label="trajectories", show_value=True)
    horizon = mo.ui.slider(200, 1200, value=600, step=200, label="steps per trajectory",
                           show_value=True)
    return horizon, n_traj, start


@app.cell
def _(C, W, grid_figure, heatmap, horizon, mo, n_traj, np, star, start):
    DS = [1, 4, 16, 64]        # four points is enough to see the trend, and 10 critics each is slow
    LABELS = [f"D={d}" for d in DS] + ["uniform"]

    # The walks, collected now; they are relabeled into batches only when training starts.
    # `uniform` is the alternative to walking at all: every state is an independent uniform draw
    # over the cells (and at training time so are the action and the goal). Its visitation here
    # is the same number of draws as a walk dataset, so the two are on the same footing.
    data = {}
    for _label, _d in zip(LABELS, DS):
        _S, _A = C.collect_outward(W, start.value, D=_d, n_traj=n_traj.value, T=horizon.value)
        data[_label] = dict(S=_S, A=_A, visits=np.bincount(_S.ravel(), minlength=W["N"]))
    _draws = np.random.default_rng(1).integers(0, W["N"], n_traj.value * (horizon.value + 1))
    data["uniform"] = dict(visits=np.bincount(_draws, minlength=W["N"]))

    _sr, _sc = W["cells"][start.value]
    _fig = grid_figure(len(LABELS), [l if l != "uniform" else "uniform (no walk)" for l in LABELS],
                       height=300)
    for _i, _label in enumerate(LABELS):
        # Each panel on its own min..max: a long walk visits every cell over a thousand times,
        # so anchoring at zero (or sharing a bar) flattens the structure that matters here.
        _v = data[_label]["visits"]
        _r, _c = heatmap(_fig, _i, W, _v, [f"cell {i}: {n} visits" for i, n in enumerate(_v)])
        if _label != "uniform":
            star(_fig, W, start.value, _r, _c)

    mo.vstack([
        mo.hstack([start, n_traj, horizon], justify="start", gap=1),
        mo.md(f"Visits per cell. {n_traj.value} walks of {horizon.value} steps, every one "
              f"starting at cell {start.value} = row {_sr}, col {_sc} (the star). D is the mean "
              f"length of an outward excursion; D=1 is the plain random walk. `uniform` does "
              f"not walk: each state is an independent uniform draw over the cells."),
        _fig,
    ])
    return DS, LABELS, data


@app.cell
def _(mo):
    steps = mo.ui.slider(1000, 30000, value=10000, step=1000, label="gradient steps",
                         show_value=True)
    # Adam's step size, for every critic. At 3e-4 the regression control has not converged by
    # the time the argmax matters: the gap between the best action and a wall-bump is
    # gamma^d * (1 - gamma) ~ 0.01-0.03, and 10k steps at 3e-4 leave the fit error above that.
    lr = mo.ui.dropdown(options={"3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3, "1e-2": 1e-2},
                        value="3e-3", label="learning rate")
    train = mo.ui.run_button(label="train 10 critics on every dataset")
    return lr, steps, train


@app.cell
def _(C, G, W, data, lr, make_batches, np, steps):
    def relabel(label):
        """Two relabelings of the same walks: geometric future goals for the contrastive critics
        (their positives must be the discounted occupancy), uniform-future goals -- HER -- for TD
        and the regression control. `uniform` has no walk."""
        if label == "uniform":
            b = C.uniform_batches(W, 1, steps.value)
            return b, b
        S, A = data[label]["S"], data[label]["A"]
        return (make_batches(W, S, A, 1, steps.value),
                make_batches(W, S, A, 1, steps.value, future="uniform"))

    def train_and_score(label, **kw):
        """Every critic on one dataset, evaluated, plus a fixed colour range per critic taken
        over several goals. Autoscaling each panel to its own min/max would render a map that
        merely shifts with the goal as *identical* pictures, which is exactly what a critic that
        barely conditions on the goal produces -- the failure would be invisible."""
        b, b_td = relabel(label)
        # Hard-max backup. The soft (SAC) backup adds alpha * entropy per step, which compounds
        # to alpha * ln4 / (1 - gamma) -- ~1.9 at alpha = 0.05 with this gamma, against a Q*
        # that tops out at 1.0. That swamps the 0.01-0.03 gap between the best action and a
        # wall-bump, and the TD critics land below chance. With the max they reach the control.
        trained = {}
        for _, out in C.train_suite(W, 1, b, b_td, lr=lr.value, kind="td3", **kw):
            trained = out
        probe_goals = np.linspace(0, W["N"] - 1, 5).astype(int)
        vrange = {}
        for k, v in trained.items():
            maps = np.concatenate([G.value_map(W, v, int(g)) for g in probe_goals])
            vrange[k] = (float(maps.min()), float(maps.max()))
        return dict(critics=trained,
                    scores={k: G.evaluate(W, v, n_eval=150, n_probe=300)
                            for k, v in trained.items()},
                    vrange=vrange)

    return (train_and_score,)


@app.cell
def _(LABELS, lr, mo, steps, train, train_and_score):
    runs = {}
    if train.value:
        with mo.status.progress_bar(total=len(LABELS), title="training every dataset") as _bar:
            for _label in LABELS:
                runs[_label] = train_and_score(_label)
                _bar.update()

    mo.vstack([
        mo.hstack([steps, lr, train], justify="start", gap=1),
        mo.md(f"Trained {len(LABELS)} datasets x 10 critics. The selector below is instant -- "
              f"nothing retrains." if runs else
              f"Each dataset is relabeled into {steps.value} batches of 256 and every critic "
              f"takes one step per batch. {len(LABELS)} datasets x 10 critics is a couple of "
              f"minutes on the GPU."),
    ])
    return (runs,)


@app.cell
def _(LABELS, W, mo, np, runs):
    mo.stop(not runs, mo.md("*(train to explore)*"))
    dataset = mo.ui.dropdown(options=LABELS, value=LABELS[0], label="dataset")
    goal_pick = mo.ui.slider(0, W["N"] - 1, value=int(np.argmax(W["dist"][:, 0])),
                             label="goal cell", show_value=True)
    return dataset, goal_pick


@app.cell
def _(dataset, go, make_subplots, mo, np, runs):
    def _figure():
        critics = runs[dataset.value]["critics"]
        scores = runs[dataset.value]["scores"]
        # One panel per critic. Eight curves on shared axes were unreadable, and the three
        # quantities involved -- a contrastive loss, a squared error and a classification
        # accuracy -- do not share a scale anyway. Hover for values, drag to zoom, double-click
        # to reset.
        cols = -(-len(critics) // 2)
        fig = make_subplots(
            rows=2, cols=cols, specs=[[{"secondary_y": True}] * cols] * 2,
            subplot_titles=[
                f"{name} -- {'contrastive' if e['accs'] else 'regression on Q*' if name.endswith('_mse') else 'TD'}"
                f"<br>argmax accuracy {scores[name]['agree']:.2f}"
                for name, e in critics.items()],
            vertical_spacing=0.16, horizontal_spacing=0.06)
        k = 50                                   # the raw loss is noisy; show a running mean too
        n = len(next(iter(critics.values()))["losses"])
        stride = max(1, n // 1000)               # ~1000 points per trace; more is megabytes of
                                                 # JSON the browser has to swallow, not detail
        smooth = lambda y: np.convolve(y, np.ones(k) / k, "valid")[::stride]
        x = np.arange(n)[k - 1::stride]
        for i, (name, e) in enumerate(critics.items()):
            r, c = divmod(i, cols)
            colour = "#2a7d4f" if e["accs"] else "#b5651d" if name.endswith("_mse") else "#3b6ea5"
            y = np.asarray(e["losses"])
            fig.add_trace(go.Scatter(x=np.arange(n)[::stride], y=y[::stride], mode="lines",
                                     line=dict(color=colour, width=0.7), opacity=0.25,
                                     hoverinfo="skip", showlegend=False),
                          row=r + 1, col=c + 1)
            fig.add_trace(go.Scatter(x=x, y=smooth(y), mode="lines",
                                     line=dict(color=colour, width=1.8),
                                     name="loss", showlegend=False,
                                     hovertemplate="step %{x}<br>loss %{y:.3g}<extra></extra>"),
                          row=r + 1, col=c + 1)
            if e["accs"]:
                # the contrastive critics also carry the accuracy they are really optimizing
                fig.add_trace(go.Scatter(x=x, y=smooth(np.asarray(e["accs"])), mode="lines",
                                         line=dict(color="#c0392b", width=1.4),
                                         name="pair accuracy", showlegend=False,
                                         hovertemplate="step %{x}<br>pair acc %{y:.3f}<extra></extra>"),
                              row=r + 1, col=c + 1, secondary_y=True)
                fig.update_yaxes(range=[0, 1.02], secondary_y=True, row=r + 1, col=c + 1,
                                 title_text="pair acc", title_font=dict(size=9, color="#c0392b"))
            else:
                fig.update_yaxes(type="log", row=r + 1, col=c + 1)
            fig.update_yaxes(title_text="loss", title_font=dict(size=9, color=colour),
                             secondary_y=False, row=r + 1, col=c + 1)
        fig.update_annotations(font_size=10)
        fig.update_layout(
            template="plotly_white", height=560, margin=dict(l=40, r=20, t=70, b=30),
            hovermode="x",
            title=dict(text=f"{dataset.value} data. Green = contrastive loss (red = its pair "
                            f"accuracy), blue = TD squared error, orange = regression squared "
                            f"error. Bold line is a {k}-step running mean.", font=dict(size=11)))
        return fig

    mo.vstack([dataset, _figure()])
    return


@app.cell
def _(ARROW, G, W, go, heatmap, np, star):
    def critic_panel(fig, panel, goal, V, qs, title_prefix, vrange, cols=None):
        """One critic, one goal: V(s, g) as colour, the greedy move as an arrow at every cell,
        black where BFS agrees that move is optimal and magenta where it does not. Hover a cell
        for the Q of all four actions. Returns the share of non-goal cells whose arrow is
        black, which also goes into the panel title."""
        opt = G.optimal_field(W, goal)                       # (N, 4) mask of optimal actions
        act = qs.argmax(1)
        cells = np.arange(W["N"]) != goal                    # no "optimal move" at the goal
        good = opt[np.arange(W["N"]), act] & cells
        share = good.sum() / cells.sum()

        hover = [f"cell {i}: V = {V[i]:.3f}<br>"
                 + "  ".join(f"{a}{q:.3f}" for a, q in zip(ARROW, qs[i]))
                 + f"<br>optimal: {''.join(ARROW[opt[i]])}" for i in range(W["N"])]
        r, c = heatmap(fig, panel, W, V, hover, colorscale="RdBu_r", zmin=vrange[0],
                       zmax=vrange[1], cols=cols)
        rows, cols_ = W["cells"][cells, 0], W["cells"][cells, 1]
        fig.add_trace(go.Scatter(
            x=cols_, y=rows, mode="text", text=ARROW[act[cells]], hoverinfo="skip",
            textfont=dict(size=15, color=np.where(good[cells], "black", "#ff2fd6"))),
            row=r, col=c)
        star(fig, W, goal, r, c)
        fig.layout.annotations[panel].text = f"{title_prefix}<br>{100 * share:.0f}% of arrows optimal"
        return share

    LEGEND = ("colour = V(s, g) = max over actions of Q. **black arrow** = greedy move is "
              "BFS-optimal, **magenta arrow** = it is not, **star** = goal. Hover a cell for "
              "the Q of every action.")
    return LEGEND, critic_panel


@app.cell
def _(G, W, critic_panel, grid_figure, np):
    # One grid: a row per objective, a column per factorization, the oracle at the left of
    # every row so each critic is read against the answer. Down a column is the objective's
    # cost with the architecture held fixed; along a row, the factorization's cost with the
    # objective held fixed. The monolithic critic has no two-tower structure, so no contrastive
    # form -- that slot stays empty. sa_g's InfoNCE is the backward direction (normalized over
    # (s, a)); see CRITICS in gridworld/coverage.py for why.
    ROWS = [("regression on the oracle", ["monolithic_mse", "sa_g_mse", "sg_a_mse"]),
            ("TD", ["monolithic_td", "sa_g_td", "sg_a_td"]),
            ("InfoNCE", [None, "sa_g_infonce", "sg_a_infonce"]),
            ("binary NCE", [None, "sa_g_binary_nce", "sg_a_binary_nce"])]
    COLS = ["optimal Q", "monolithic", "sa_g  (CRL)", "sg_a  (CARL)"]

    def critic_grid(entry, goal, title):
        states, goals = np.arange(W["N"]), np.full(W["N"], goal)
        q_star = np.stack([G.optimal_q(W, states, goals, np.repeat(c[None], W["N"], 0))
                           for c in G.all_chunks(1)], 1)
        finite = W["dist"][np.isfinite(W["dist"])]
        oracle_range = (float(W["gamma"] ** finite.max()), 1.0)

        fig = grid_figure(len(ROWS) * len(COLS), [" "] * (len(ROWS) * len(COLS)), height=270,
                          cols=len(COLS))
        for r, (objective, names) in enumerate(ROWS):
            critic_panel(fig, r * len(COLS), goal, q_star.max(1), q_star, COLS[0], oracle_range,
                         cols=len(COLS))
            for c, name in enumerate(names, start=1):
                panel = r * len(COLS) + c
                if name is None:
                    fig.layout.annotations[panel].text = f"{COLS[c]}<br>(no contrastive form)"
                    continue
                e = entry["critics"][name]
                qs = np.asarray(G.q_over_chunks(W, e, states, goals))
                critic_panel(fig, panel, goal, qs.max(1), qs, COLS[c], entry["vrange"][name],
                             cols=len(COLS))
            # the row's objective, down the left margin
            yd = fig.get_subplot(r + 1, 1).yaxis.domain
            fig.add_annotation(x=-0.01, y=(yd[0] + yd[1]) / 2, xref="paper", yref="paper",
                               text=f"<b>{objective}</b>", textangle=-90, showarrow=False,
                               font=dict(size=12), xanchor="right")
        fig.update_layout(margin=dict(l=40, t=90), title=dict(text=title, font=dict(size=11)))
        return fig

    return (critic_grid,)


@app.cell
def _(LEGEND, critic_grid, dataset, goal_pick, mo, runs):
    mo.vstack([
        mo.hstack([dataset, goal_pick], justify="start", gap=1),
        critic_grid(runs[dataset.value], int(goal_pick.value),
                    f"{dataset.value} data, goal at cell {goal_pick.value}. Colour range is "
                    f"fixed per critic across goals, so a map that does not move is a critic "
                    f"that ignores the goal."),
        mo.md(LEGEND),
    ])
    return


@app.cell
def _(C, DS, go, mo, runs):
    mo.stop(not runs, mo.md(""))

    def _figure():
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                   "#7f7f7f", "#bcbd22", "#17becf", "#393b79"]
        xs = [str(d) for d in DS]
        fig = go.Figure()
        for (name, _, _), colour in zip(C.CRITICS, palette):
            # click a legend entry to hide it, double-click to isolate it; the dotted line is
            # the same critic on uniform data and toggles with it
            fig.add_trace(go.Scatter(
                x=xs, y=[runs[f"D={d}"]["scores"][name]["agree"] for d in DS],
                mode="lines+markers", name=name, legendgroup=name,
                line=dict(color=colour, dash="dash" if name.endswith("_mse") else "solid"),
                hovertemplate="D=%{x}: %{y:.2f}<extra>" + name + "</extra>"))
            fig.add_trace(go.Scatter(
                x=xs, y=[runs["uniform"]["scores"][name]["agree"]] * len(DS),
                mode="lines", legendgroup=name, showlegend=False, opacity=0.5,
                line=dict(color=colour, dash="dot", width=1),
                hovertemplate="uniform: %{y:.2f}<extra>" + name + "</extra>"))
        fig.update_layout(
            template="plotly_white", height=420, margin=dict(l=40, r=20, t=60, b=40),
            hovermode="x unified",
            title=dict(text="share of cells where the critic picks a BFS-optimal move "
                            "(dotted = the same critic trained on uniform data)",
                       font=dict(size=12)),
            xaxis_title="effective excursion length D", yaxis=dict(range=[0, 1], title="accuracy"),
            legend=dict(font=dict(size=10)))
        return fig

    _figure()
    return


@app.cell
def _(LABELS, mo):
    # The same critics at five network sizes, on one dataset. Width is every MLP's hidden width;
    # the two-tower embedding scales with it (width / 4, so 128 -> 32, today's default).
    WIDTHS = [32, 64, 128, 256, 512]
    scale_dataset = mo.ui.dropdown(options=LABELS, value=LABELS[1], label="dataset")
    scale_train = mo.ui.run_button(label="train 10 critics at 5 sizes")
    return WIDTHS, scale_dataset, scale_train


@app.cell
def _(WIDTHS, mo, scale_dataset, scale_train, train_and_score):
    by_size = {}
    if scale_train.value:
        with mo.status.progress_bar(total=len(WIDTHS), title="training every size") as _bar:
            for _w in WIDTHS:
                by_size[_w] = train_and_score(scale_dataset.value, width=_w, repr_dim=_w // 4)
                _bar.update()
    return (by_size,)


@app.cell
def _(C, WIDTHS, by_size, go, make_subplots, mo, scale_dataset, scale_train):
    mo.stop(not by_size, mo.vstack([mo.hstack([scale_dataset, scale_train], justify="start",
                                              gap=1),
                                    mo.md("*(train across sizes to see how accuracy scales)*")]))

    def _figure():
        names = [n for n, _, _ in C.CRITICS]
        cols = -(-len(names) // 2)
        fig = make_subplots(rows=2, cols=cols, subplot_titles=names, vertical_spacing=0.18,
                            horizontal_spacing=0.05, shared_yaxes=True)
        for i, name in enumerate(names):
            r, c = divmod(i, cols)
            colour = ("#b5651d" if name.endswith("_mse") else
                      "#2a7d4f" if "nce" in name else "#3b6ea5")
            fig.add_trace(go.Scatter(
                x=[by_size[w]["critics"][name]["n_params"] for w in WIDTHS],
                y=[by_size[w]["scores"][name]["agree"] for w in WIDTHS],
                mode="lines+markers", line=dict(color=colour), showlegend=False,
                text=[f"width {w}" for w in WIDTHS],
                hovertemplate="%{text}: %{x:,} params<br>accuracy %{y:.2f}<extra></extra>"),
                row=r + 1, col=c + 1)
            fig.update_xaxes(type="log", row=r + 1, col=c + 1, tickfont=dict(size=8))
            fig.update_yaxes(range=[0, 1], row=r + 1, col=c + 1, tickfont=dict(size=8))
        fig.update_annotations(font_size=10)
        fig.update_layout(template="plotly_white", height=480,
                          margin=dict(l=40, r=20, t=70, b=40),
                          title=dict(text=f"argmax accuracy against parameter count, "
                                          f"{scale_dataset.value} data", font=dict(size=12)),
                          xaxis_title="parameters")
        return fig

    mo.vstack([mo.hstack([scale_dataset, scale_train], justify="start", gap=1), _figure()])
    return


@app.cell
def _(W, WIDTHS, by_size, mo, np):
    mo.stop(not by_size, mo.md(""))
    scale_width = mo.ui.dropdown(options={str(w): w for w in WIDTHS}, value="128", label="width")
    scale_goal = mo.ui.slider(0, W["N"] - 1, value=int(np.argmax(W["dist"][:, 0])),
                              label="goal cell", show_value=True)
    return scale_goal, scale_width


@app.cell
def _(LEGEND, by_size, critic_grid, mo, scale_dataset, scale_goal, scale_width):
    mo.vstack([
        mo.hstack([scale_dataset, scale_width, scale_goal], justify="start", gap=1),
        critic_grid(by_size[scale_width.value], int(scale_goal.value),
                    f"{scale_dataset.value} data, width {scale_width.value}, goal at cell "
                    f"{scale_goal.value}."),
        mo.md(LEGEND),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Reading `uniform`

    `uniform` is not simply "more data". It draws the goal uniformly over the maze, and the
    objective families care about that differently.

    A **contrastive** critic is fit to say its goal *is* a discounted future of its
    (state, action). A uniformly drawn goal replaces that target with a different one, so the
    critic no longer estimates the discounted occupancy and the policy read off it means
    nothing -- its collapse there is a broken objective, not a data shortage.

    A **TD** critic has no such assumption; the Bellman backup is indifferent to where the
    goal came from so long as the pairs are covered. Its decline is the ordinary one: the same
    capacity now has to fit every distance, and accuracy near the goal -- which is what the
    greedy action depends on -- is what gets spent.

    The **regression** critics settle which of the two is happening. Same data, same
    architecture, exact targets: where they stay high and the RL objectives fall, the data was
    never the problem.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Why the ceiling is not 1.0

    The regression critics are handed the exact optimal Q and still only reach ~0.6 here, at a
    mean squared error of ~5e-4. Those are consistent: with a half-life gamma the values of
    two actions at distance 30 differ by a few thousandths, so a loss that is already tiny has
    almost nothing left to say about which of them is better. Squared error on values and
    correctness of the argmax are different objectives, and on a long-horizon maze they come
    apart -- which is worth remembering when reading any of the loss curves below.

    The gap between a critic and its regression counterpart is the part attributable to the RL
    objective. The gap between the regression counterpart and 1.0 is the part attributable to
    the architecture, the horizon, and the budget.
    """)
    return


if __name__ == "__main__":
    app.run()
