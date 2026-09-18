"""The coverage study, made continuous: how much of a critic's accuracy is the objective, and how
much is the data, when the state is a position and the action a displacement.

Same design as `coverage_study.py`. Every critic is trained on relabeled trajectories from a
noisy waypoint navigator that heads for random points along the geodesic and picks another on
arrival; the dial D is the radius around the start its waypoints are drawn from, so D = 1 is a
blob around the start and D = 64 the whole world (`uniform` draws state, action and goal
independently). Critics are scored against the geodesic
distance around the walls: a greedy action counts as a good pick when it makes at least half
the progress toward the goal that the best of the candidate actions would have made.

What the continuous world adds is *topology*. A perfect maze is a tree -- one path between any
two points -- and on a tree the action that most increases a random walk's chance of reaching
the goal is also the optimal one, so a critic that learns the behaviour policy's occupancy
(which is what a contrastive critic learns) cannot be told apart from one that learns Q*. The
braided maze opens dead ends into loops, and the rooms world is nothing but loops. There the
two come apart, and the study asks which critics notice.

Ten critics: the three factorizations under TD, the two two-tower ones under InfoNCE and binary
NCE, and all three regressed directly on the optimal Q -- the control that separates what the
data allows from what the objective delivers. The max over a continuous action space is a
sampled max over a fixed pool of candidate chunks, as QT-Opt does, so no actor is needed.
Positions enter every network through sin/cos features (`pointmaze/critics.py`), the continuous
analogue of the gridworld's one-hot: with raw coordinates the regression control tops out near
0.5, because a value function that jumps across a wall is not something an MLP of (x, y) can
bend to. Even so, the control is capped by fit precision -- an oracle corrupted with the
control's own error scores the same -- so read every critic against the control, not against 1.

Each knob sits next to the picture it changes. The world and the walks are rebuilt the moment a
slider moves; training is behind a button.

Run it with:  uv run --with marimo --with plotly marimo edit notebooks/pointmaze_coverage.py
"""

import marimo

__generated_with = "0.24.2"
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

    import pointmaze as P
    from pointmaze import coverage as C

    return C, P, go, make_subplots, mo, np


@app.cell
def _(go, make_subplots, np):
    # Every picture of the world is a heatmap over free space with the walls drawn on top, one
    # panel per thing compared. Coordinates are world units: x is the column, y the row, row 0
    # at the top.
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
        fig.update_yaxes(visible=False, autorange="reversed")
        for i in range(n_panels):                                       # square cells
            fig.update_yaxes(scaleanchor="x" if i == 0 else f"x{i + 1}", scaleratio=1,
                             row=i // cols + 1, col=i % cols + 1)
        return fig

    def heatmap(fig, panel, W, img, hover, colorscale="Viridis", zmin=None, zmax=None,
                cols=None):
        """One panel: `img` over the fine grid (NaN in walls), `hover` a per-node string of the
        same shape, the walls in grey, a colourbar beside the panel. `panel` is 0-based."""
        cols = cols or len(fig._grid_ref[0])
        r, c = panel // cols + 1, panel % cols + 1
        sub = fig.get_subplot(r, c)
        xd, yd = sub.xaxis.domain, sub.yaxis.domain
        H = img.shape[0]
        xs = (np.arange(H) + 0.5) * (W["n"] / H)
        fig.add_trace(go.Heatmap(
            z=img, x=xs, y=xs, text=hover, hoverongaps=False,
            hovertemplate="%{text}<extra></extra>", colorscale=colorscale, zmin=zmin, zmax=zmax,
            colorbar=dict(x=xd[1] + 0.004, y=(yd[0] + yd[1]) / 2, len=0.7 * (yd[1] - yd[0]),
                          thickness=8, tickfont=dict(size=8), xpad=0)),
            row=r, col=c)
        cx = np.arange(W["n"]) + 0.5
        fig.add_trace(go.Heatmap(
            z=np.where(W["walls"], 1.0, np.nan), x=cx, y=cx, hoverinfo="skip", showscale=False,
            colorscale=[[0, "#3a3a3a"], [1, "#3a3a3a"]]), row=r, col=c)
        return r, c

    def star(fig, W, pos, r, c):
        fig.add_trace(go.Scatter(x=[pos[1]], y=[pos[0]], mode="markers", hoverinfo="skip",
                                 marker=dict(symbol="star", size=13, color="gold",
                                             line=dict(color="black", width=1))), row=r, col=c)

    def arrows(fig, pos, vec, good, r, c):
        """A unit vector at every position, as an arrow marker: black where `good`, magenta
        where not. Angle is clockwise from north; row increases downward."""
        ang = np.degrees(np.arctan2(vec[:, 1], -vec[:, 0]))
        for mask, colour in ((good, "black"), (~good, "#ff2fd6")):
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=pos[mask, 1], y=pos[mask, 0], mode="markers", hoverinfo="skip",
                    marker=dict(symbol="arrow", angle=ang[mask], size=9, color=colour,
                                line=dict(color="white", width=0.5))), row=r, col=c)

    return arrows, grid_figure, heatmap, star


@app.cell
def _(mo):
    kind = mo.ui.dropdown(options=["maze", "braid", "rooms"], value="braid", label="topology")
    size = mo.ui.slider(9, 21, value=15, step=2, label="size", show_value=True)
    seed = mo.ui.slider(0, 5, value=0, label="seed", show_value=True)
    return kind, seed, size


@app.cell
def _(P, grid_figure, heatmap, kind, mo, np, seed, size):
    # `maze` is a tree; `braid` is the same maze with 30% of its dead ends opened into loops;
    # `rooms` is a 3x3 grid of rooms with one door per shared wall. Critics see raw (x, y),
    # scaled to [0, 1] -- there is no tabular option once the state is continuous, so the
    # network has to infer the walls from position, and the regression control shows how well
    # a given size can.
    W = P.build_world(kind.value, size.value, seed=seed.value)

    _fig = grid_figure(1, [""], height=300)
    _fig.update_layout(width=300, margin=dict(l=10, r=10, t=10, b=10))
    _img = np.where(W["walls"], np.nan, 0.0)
    _hover = np.full(W["walls"].shape, "", dtype=object)
    for _i, (_r, _c) in enumerate(W["free_cells"]):
        _hover[_r, _c] = f"cell {_i} (row {_r}, col {_c})"
    heatmap(_fig, 0, W, _img, _hover, colorscale=[[0, "#e8e8e8"], [1, "#e8e8e8"]])
    _fig.data[0].showscale = False

    _d = W["fine_dist"][np.isfinite(W["fine_dist"])]
    mo.vstack([
        mo.hstack([kind, size, seed], justify="start", gap=1),
        mo.hstack([_fig, mo.md(
            f"**{kind.value}**, {W['n']}x{W['n']}, **{W['N']}** open cells of unit width; a step "
            f"moves up to {W['step']} per axis. Mean geodesic distance **{_d.mean():.1f}** units = "
            f"**{_d.mean() / W['step']:.0f}** steps, longest **{_d.max() / W['step']:.0f}** steps. "
            f"gamma **{W['gamma']:.3f}**, the half-life of one mean path. Hover a cell for its "
            f"index -- the `start cell` and `goal cell` sliders take indices."
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
def _(C, P, W, grid_figure, heatmap, horizon, mo, n_traj, np, star, start):
    DS = [1, 4, 16, 64]
    LABELS = [f"D={d}" for d in DS] + ["uniform"]

    def _visits(pos):
        """Visits per cell, as an image. Per cell rather than per fine-grid node: the outward
        moves are half-cell compass steps from a cell centre, so positions sit on a lattice
        that a finer histogram renders as a checkerboard."""
        cell = np.clip(np.floor(pos.reshape(-1, 2)).astype(int), 0, W["n"] - 1)
        img = np.zeros((W["n"], W["n"]))
        np.add.at(img, (cell[:, 0], cell[:, 1]), 1)
        return np.where(W["walls"], np.nan, img)

    # The walks, collected now; they are relabeled into batches only when training starts.
    # `uniform` does not walk: each state is an independent uniform draw over free space.
    data = {}
    for _label, _d in zip(LABELS, DS):
        _P, _A = C.collect_outward(W, start.value, D=_d, n_traj=n_traj.value, T=horizon.value)
        data[_label] = dict(P=_P, A=_A, visits=_visits(_P))
    data["uniform"] = dict(visits=_visits(P.sample_positions(
        W, n_traj.value * (horizon.value + 1), np.random.default_rng(1))))

    _fig = grid_figure(len(LABELS), [l if l != "uniform" else "uniform (no walk)" for l in LABELS],
                       height=300)
    for _i, _label in enumerate(LABELS):
        _v = data[_label]["visits"]
        _r, _c = heatmap(_fig, _i, W, _v, np.where(np.isnan(_v), "",
                                                    np.nan_to_num(_v).astype(int).astype(str)))
        if _label != "uniform":
            star(_fig, W, P.cell_centre(W, start.value), _r, _c)

    _sr, _sc = W["free_cells"][start.value]
    mo.vstack([
        mo.hstack([start, n_traj, horizon], justify="start", gap=1),
        mo.md(f"Visits per cell. {n_traj.value} walks of {horizon.value} steps, every one "
              f"starting at the centre of cell {start.value} = row {_sr}, col {_sc} (the star). "
              f"The walker navigates to random waypoints along the geodesic, with a quarter of "
              f"its steps random; D is the radius around the start, in cells, its waypoints are "
              f"drawn from."),
        _fig,
    ])
    return DS, LABELS, data


@app.cell
def _(mo):
    steps = mo.ui.slider(1000, 30000, value=10000, step=1000, label="gradient steps",
                         show_value=True)
    lr = mo.ui.dropdown(options={"3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3}, value="1e-3",
                        label="learning rate")
    n_cand = mo.ui.dropdown(options={"64": 64, "256": 256}, value="64", label="candidate actions")
    train = mo.ui.run_button(label="train 10 critics on every dataset")
    return lr, n_cand, steps, train


@app.cell
def _(C, P, W, data, lr, n_cand, np, steps):
    def relabel(label):
        """Geometric future goals for the contrastive critics, uniform-future goals (HER) for TD
        and the regression control. `uniform` has no walk."""
        if label == "uniform":
            b = C.uniform_batches(W, 1, steps.value)
            return b, b
        Pp, A = data[label]["P"], data[label]["A"]
        return (P.make_batches(W, Pp, A, 1, steps.value),
                P.make_batches(W, Pp, A, 1, steps.value, future="uniform"))

    def train_and_score(label, **kw):
        """Every critic on one dataset, evaluated, plus a fixed colour range per critic taken
        over several goals (autoscaling would hide a critic that ignores the goal)."""
        b, b_td = relabel(label)
        trained = {}
        for _, out in C.train_suite(W, 1, b, b_td, cand=P.candidates(1, n_cand.value),
                                    lr=lr.value, kind="td3", **kw):
            trained = out
        probe_goals = P.sample_positions(W, 5, np.random.default_rng(3))
        vrange = {}
        for k, v in trained.items():
            maps = np.concatenate([P.value_grid(W, v, g).ravel() for g in probe_goals])
            vrange[k] = (float(np.nanmin(maps)), float(np.nanmax(maps)))
        return dict(critics=trained,
                    scores={k: P.evaluate(W, v, n_eval=150, n_probe=300)
                            for k, v in trained.items()},
                    vrange=vrange)

    return (train_and_score,)


@app.cell
def _(LABELS, lr, mo, n_cand, steps, train, train_and_score):
    runs = {}
    if train.value:
        with mo.status.progress_bar(total=len(LABELS), title="training every dataset") as _bar:
            for _label in LABELS:
                runs[_label] = train_and_score(_label)
                _bar.update()

    mo.vstack([
        mo.hstack([steps, lr, n_cand, train], justify="start", gap=1),
        mo.md(f"Trained {len(LABELS)} datasets x 10 critics. The selector below is instant -- "
              f"nothing retrains." if runs else
              f"Each dataset is relabeled into {steps.value} batches of 256 and every critic "
              f"takes one step per batch. The max over actions is over {n_cand.value} fixed "
              f"candidates. {len(LABELS)} datasets x 10 critics is a few minutes on the GPU."),
    ])
    return (runs,)


@app.cell
def _(LABELS, W, mo, np, runs):
    mo.stop(not runs, mo.md("*(train to explore)*"))
    dataset = mo.ui.dropdown(options=LABELS, value=LABELS[1], label="dataset")
    goal_pick = mo.ui.slider(0, W["N"] - 1, value=int(np.argmax(W["cell_dist"][:, 0])),
                             label="goal cell", show_value=True)
    return dataset, goal_pick


@app.cell
def _(dataset, go, make_subplots, mo, np, runs):
    def _figure():
        critics = runs[dataset.value]["critics"]
        scores = runs[dataset.value]["scores"]
        cols = -(-len(critics) // 2)
        fig = make_subplots(
            rows=2, cols=cols, specs=[[{"secondary_y": True}] * cols] * 2,
            subplot_titles=[
                f"{name} -- {'contrastive' if e['accs'] else 'regression on Q*' if name.endswith('_mse') else 'TD'}"
                f"<br>accuracy {scores[name]['agree']:.2f}"
                for name, e in critics.items()],
            vertical_spacing=0.16, horizontal_spacing=0.06)
        k = 50
        n = len(next(iter(critics.values()))["losses"])
        stride = max(1, n // 1000)
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
                                     line=dict(color=colour, width=1.8), showlegend=False,
                                     hovertemplate="step %{x}<br>loss %{y:.3g}<extra></extra>"),
                          row=r + 1, col=c + 1)
            if e["accs"]:
                fig.add_trace(go.Scatter(x=x, y=smooth(np.asarray(e["accs"])), mode="lines",
                                         line=dict(color="#c0392b", width=1.4), showlegend=False,
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
def _(P, W, arrows, grid_figure, heatmap, np, star):
    # One grid: a row per objective, a column per factorization, the geodesic at the left of
    # every row. Colour is V(s, g) = max over candidate chunks on the fine grid; the arrow at
    # every cell centre is the greedy action, black when it is a good pick (at least half the
    # progress of the best candidate) and magenta when not. The monolithic critic has no
    # two-tower structure, so no contrastive form.
    ROWS = [("regression on the oracle", ["monolithic_mse", "sa_g_mse", "sg_a_mse"]),
            ("TD", ["monolithic_td", "sa_g_td", "sg_a_td"]),
            ("InfoNCE", [None, "sa_g_infonce", "sg_a_infonce"]),
            ("binary NCE", [None, "sa_g_binary_nce", "sg_a_binary_nce"])]
    COLS = ["geodesic", "monolithic", "sa_g  (CRL)", "sg_a  (CARL)"]
    LEGEND = ("colour = V(s, g) = max over candidate actions. **black arrow** = the greedy action "
              "makes at least half the progress of the best candidate, **magenta arrow** = it "
              "does not, **star** = goal.")

    def _my_hover(img, label):
        return np.where(np.isnan(img), "", np.char.add(f"{label} = ", np.round(img, 3).astype(str)))

    def critic_grid(entry, goal_cell, title):
        goal = P.cell_centre(W, goal_cell)
        pos = P.cell_grid(W)
        fig = grid_figure(len(ROWS) * len(COLS), [" "] * (len(ROWS) * len(COLS)), height=270,
                          cols=len(COLS))
        geo = geo_img = P.geodesic_grid(W, goal)
        geo_v = W["gamma"] ** (geo_img / W["step"])          # V* = gamma^(steps to go)
        for r, (objective, names) in enumerate(ROWS):
            panel = r * len(COLS)
            rr, cc = heatmap(fig, panel, W, geo_v, _my_hover(geo_v, "V*"), colorscale="RdBu_r",
                             zmin=float(np.nanmin(geo_v)), zmax=1.0, cols=len(COLS))
            arrows(fig, pos, P.ideal_field(W, goal), np.ones(len(pos), bool), rr, cc)
            star(fig, W, goal, rr, cc)
            fig.layout.annotations[panel].text = f"{COLS[0]}<br>descent direction"
            for c, name in enumerate(names, start=1):
                panel = r * len(COLS) + c
                if name is None:
                    fig.layout.annotations[panel].text = f"{COLS[c]}<br>(no contrastive form)"
                    continue
                e = entry["critics"][name]
                v = P.value_grid(W, e, goal)
                lo, hi = entry["vrange"][name]
                rr, cc = heatmap(fig, panel, W, v, _my_hover(v, "V"), colorscale="RdBu_r",
                                 zmin=lo, zmax=hi, cols=len(COLS))
                vec, good = P.action_field(W, e, goal)
                arrows(fig, pos, vec, good, rr, cc)
                star(fig, W, goal, rr, cc)
                fig.layout.annotations[panel].text = (
                    f"{COLS[c]}<br>{100 * good.mean():.0f}% good picks")
            yd = fig.get_subplot(r + 1, 1).yaxis.domain
            fig.add_annotation(x=-0.01, y=(yd[0] + yd[1]) / 2, xref="paper", yref="paper",
                               text=f"<b>{objective}</b>", textangle=-90, showarrow=False,
                               font=dict(size=12), xanchor="right")
        fig.update_layout(margin=dict(l=40, t=90), title=dict(text=title, font=dict(size=11)))
        return fig

    return LEGEND, critic_grid


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
def _(C, DS, P, W, go, mo, n_cand, runs):
    mo.stop(not runs, mo.md(""))

    def _figure():
        floor = P.random_floor(W, P.candidates(1, n_cand.value))
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                   "#7f7f7f", "#bcbd22", "#17becf"]
        xs = [str(d) for d in DS]
        fig = go.Figure()
        for (name, _, _), colour in zip(C.CRITICS, palette):
            fig.add_trace(go.Scatter(
                x=xs, y=[runs[f"D={d}"]["scores"][name]["agree"] for d in DS],
                customdata=[runs[f"D={d}"]["scores"][name]["success"] for d in DS],
                mode="lines+markers", name=name, legendgroup=name,
                line=dict(color=colour, dash="dash" if name.endswith("_mse") else "solid"),
                hovertemplate="D=%{x}: %{y:.2f} (rollout success %{customdata:.2f})"
                              "<extra>" + name + "</extra>"))
            fig.add_trace(go.Scatter(
                x=xs, y=[runs["uniform"]["scores"][name]["agree"]] * len(DS),
                mode="lines", legendgroup=name, showlegend=False, opacity=0.5,
                line=dict(color=colour, dash="dot", width=1),
                hovertemplate="uniform: %{y:.2f}<extra>" + name + "</extra>"))
        fig.add_hline(y=floor, line=dict(color="grey", dash="dot", width=1),
                      annotation_text="random candidate", annotation_font_size=9)
        fig.update_layout(
            template="plotly_white", height=420, margin=dict(l=40, r=20, t=60, b=40),
            hovermode="x unified",
            title=dict(text="share of (position, goal) probes where the greedy action makes at "
                            "least half the best candidate's progress (dotted = the same critic "
                            "on uniform data)", font=dict(size=12)),
            xaxis_title="effective excursion length D", yaxis=dict(range=[0, 1], title="accuracy"),
            legend=dict(font=dict(size=10)))
        return fig

    _figure()
    return


@app.cell
def _(LABELS, mo):
    # The same critics at five network sizes, on one dataset. Width is every MLP's hidden width;
    # the two-tower embedding scales with it (width / 4, so 256 -> 64, today's default).
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
                          title=dict(text=f"accuracy against parameter count, "
                                          f"{scale_dataset.value} data", font=dict(size=12)),
                          xaxis_title="parameters")
        return fig

    mo.vstack([mo.hstack([scale_dataset, scale_train], justify="start", gap=1), _figure()])
    return


@app.cell
def _(W, WIDTHS, by_size, mo, np):
    mo.stop(not by_size, mo.md(""))
    scale_width = mo.ui.dropdown(options={str(w): w for w in WIDTHS}, value="256", label="width")
    scale_goal = mo.ui.slider(0, W["N"] - 1, value=int(np.argmax(W["cell_dist"][:, 0])),
                              label="goal cell", show_value=True)
    return scale_goal, scale_width


@app.cell
def _(
    LEGEND,
    by_size,
    critic_grid,
    mo,
    scale_dataset,
    scale_goal,
    scale_width,
):
    mo.vstack([
        mo.hstack([scale_dataset, scale_width, scale_goal], justify="start", gap=1),
        critic_grid(by_size[scale_width.value], int(scale_goal.value),
                    f"{scale_dataset.value} data, width {scale_width.value}, goal at cell "
                    f"{scale_goal.value}."),
        mo.md(LEGEND),
    ])
    return


if __name__ == "__main__":
    app.run()
