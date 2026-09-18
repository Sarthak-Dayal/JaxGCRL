"""Compare the Q landscapes CRL and CARL-CRL learn on ant.

Both factor the critic, but on different axes:

    CRL       Q(s, g, a)      = energy( f([s; a]) ,  g(goal) )
    CARL-CRL  Q(s, g, a_tau)  = energy( phi([s; g]) , psi(a_tau) )

CRL puts the action on the state side and the goal alone on the other; CARL moves the goal to the
state side and gives the action sequence its own encoder. The question here is what that does to
the shape of Q: hold a state and a goal fixed, sweep the action, and look at the surface each
critic paints over it.

Run it with:  uv run --with marimo marimo edit notebooks/q_landscape.py
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # Small grid evaluations; keep off the GPU so a running sweep is undisturbed.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import pickle
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from brax.io import model

    from jaxgcrl.utils.env import create_env

    return Path, create_env, jax, jnp, mo, model, np, pickle, plt


@app.cell
def _(create_env, jax):
    env = create_env("ant")
    STATE_DIM = env.state_dim
    GOAL_IDX = tuple(env.goal_indices)
    ACTION_SIZE = env.action_size

    # A handful of reset states, held fixed for every plot below. The raw env resets one key at a
    # time; the training code gets batching from its wrapper.
    _obs = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), 6)).obs
    states, goals = _obs[:, :STATE_DIM], _obs[:, STATE_DIM:]
    return ACTION_SIZE, GOAL_IDX, STATE_DIM, env, goals, states


@app.cell
def _(ACTION_SIZE, Path, mo, pickle):
    def load_run(run_dir):
        """The agent config a run was launched with, alongside its final params."""
        with open(run_dir / "args.pkl", "rb") as f:
            cfg = pickle.load(f)
        return cfg["agent"], cfg["run"]

    # Every run.py invocation writes runs/run_<exp_name>_s_<seed>/{args.pkl,ckpt/final}.
    RUNS = sorted(p for p in Path("runs").glob("run_*ant*_s_*")
                  if (p / "ckpt" / "final").exists() and (p / "args.pkl").exists()
                  and "u_maze" not in p.name)
    _labels = [p.name.replace("run_", "").replace("__ant", "") for p in RUNS]
    _crl = {l: p for l, p in zip(_labels, RUNS) if l.startswith("crl")}
    _carl = {l: p for l, p in zip(_labels, RUNS) if l.startswith("carl_crl")}

    crl_pick = mo.ui.dropdown(options=_crl, label="CRL", value=next(iter(_crl), None))
    carl_pick = mo.ui.dropdown(options=_carl, label="CARL-CRL", value=next(iter(_carl), None))
    dim_x = mo.ui.dropdown(options={f"a[{i}]": i for i in range(ACTION_SIZE)}, value="a[0]", label="x")
    dim_y = mo.ui.dropdown(options={f"a[{i}]": i for i in range(ACTION_SIZE)}, value="a[1]", label="y")
    which = mo.ui.slider(0, 5, value=0, label="state", show_value=True)
    grid_n = mo.ui.slider(11, 61, value=31, step=10, label="grid", show_value=True)
    chunk_mode = mo.ui.radio(
        options=["hold the action for the whole chunk", "vary the first action, zero the rest"],
        value="hold the action for the whole chunk", label="chunk fill (CARL only)")

    # One control block, directly above the plots it drives.
    mo.vstack([mo.hstack([crl_pick, carl_pick, which, dim_x, dim_y, grid_n],
                         justify="start", gap=1, wrap=True), chunk_mode])
    return RUNS, carl_pick, chunk_mode, crl_pick, dim_x, dim_y, grid_n, load_run, which


@app.cell
def _(carl_pick, crl_pick, jnp, load_run, model):
    def crl_q_fn(run_dir):
        """Q(s, g, a) = energy(f([s; a]), g(goal))."""
        from jaxgcrl.agents.crl.losses import energy_fn
        from jaxgcrl.agents.crl.networks import Encoder

        agent, _ = load_run(run_dir)
        _, _, critic = model.load_params(str(run_dir / "ckpt" / "final"))
        kw = dict(network_width=agent.h_dim, network_depth=agent.n_hidden,
                  skip_connections=agent.skip_connections, use_relu=agent.use_relu,
                  use_ln=agent.use_ln)
        sa_enc, g_enc = Encoder(repr_dim=agent.repr_dim, **kw), Encoder(repr_dim=agent.repr_dim, **kw)

        def q(state, goal, action):
            return energy_fn(agent.energy_fn,
                             sa_enc.apply(critic["sa_encoder"], jnp.concatenate([state, action], -1)),
                             g_enc.apply(critic["g_encoder"], goal))

        return q, agent

    def carl_q_fn(run_dir):
        """Q(s, g, a_tau) = energy(phi([s; g]), psi(a_tau))."""
        from jaxgcrl.agents.carl_crl.losses import energy_fn, psi_input
        from jaxgcrl.agents.carl_crl.networks import Encoder

        agent, _ = load_run(run_dir)
        _, _, critic = model.load_params(str(run_dir / "ckpt" / "final"))
        kw = dict(network_width=agent.h_dim, network_depth=agent.n_hidden,
                  skip_connections=agent.skip_connections, use_relu=agent.use_relu,
                  use_ln=agent.use_ln)
        sg_enc, a_enc = Encoder(repr_dim=agent.repr_dim, **kw), Encoder(repr_dim=agent.repr_dim, **kw)
        cfg = {"action_seq_len": agent.action_seq_len}

        def q(state, goal, action_seq):
            n = jnp.full((action_seq.shape[0],), agent.action_seq_len)
            return energy_fn(agent.energy_fn,
                             sg_enc.apply(critic["sg_encoder"], jnp.concatenate([state, goal], -1)),
                             a_enc.apply(critic["a_encoder"], psi_input(cfg, action_seq, n)))

        return q, agent

    crl_q, crl_cfg = crl_q_fn(crl_pick.value)
    carl_q, carl_cfg = carl_q_fn(carl_pick.value)
    return carl_cfg, carl_q, carl_q_fn, crl_cfg, crl_q, crl_q_fn


@app.cell
def _(ACTION_SIZE, carl_cfg, carl_q, chunk_mode, crl_q, dim_x, dim_y, goals,
      grid_n, jnp, np, plt, states, which):
    def _figure():
        hold = chunk_mode.value.startswith("hold")
        s = states[which.value:which.value + 1]
        g = goals[which.value:which.value + 1]

        def action_grid(seq_len):
            """A grid over two action dims, the rest at zero.

            For a chunk, `hold` repeats the action across all slots rather than leaving the tail
            at zero -- zero is a legal action, so a zero tail is a specific chunk, not a neutral one.
            """
            axis = np.linspace(-1, 1, grid_n.value)
            gx, gy = np.meshgrid(axis, axis, indexing="ij")
            a = np.zeros((axis.size**2, ACTION_SIZE))
            a[:, dim_x.value] = gx.ravel()
            a[:, dim_y.value] = gy.ravel()
            a = jnp.asarray(a)
            if seq_len == 1:
                return a, axis
            if hold:
                return jnp.tile(a, (1, seq_len)), axis
            return jnp.concatenate([a, jnp.zeros((a.shape[0], ACTION_SIZE * (seq_len - 1)))], -1), axis

        panels = []
        for name, qfn, seq_len in [("CRL", crl_q, 1),
                                   (f"CARL-CRL (n={carl_cfg.action_seq_len})", carl_q,
                                    carl_cfg.action_seq_len)]:
            acts, axis = action_grid(seq_len)
            q = qfn(jnp.repeat(s, acts.shape[0], 0), jnp.repeat(g, acts.shape[0], 0), acts)
            panels.append((name, np.asarray(q).reshape(axis.size, -1), axis))

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
        for ax, (name, q, axis) in zip(axes, panels):
            im = ax.pcolormesh(axis, axis, q.T, shading="auto", cmap="viridis")
            ax.set_xlabel(f"a[{dim_x.value}]"); ax.set_ylabel(f"a[{dim_y.value}]")
            ax.set_title(f"{name}\nQ range [{q.min():.2f}, {q.max():.2f}]")
            j, i = np.unravel_index(np.argmax(q), q.shape)
            ax.plot(axis[j], axis[i], "r*", ms=14, mec="white")
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"Q over two action dims, state {which.value} "
                     "(other dims 0, red star = argmax)")
        return fig

    _figure()
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Q over the goal, for a fixed state and action

        The same critics swept the other way: hold the state and the action, move the goal over the
        arena. This is the map a goal-conditioned policy is actually steered by.
        """
    )
    return


@app.cell
def _(ACTION_SIZE, GOAL_IDX, carl_cfg, carl_q, crl_q, jnp, np, plt, states, which):
    def _figure():
        s = states[which.value:which.value + 1]
        span = np.linspace(-12, 12, 41)
        gx, gy = np.meshgrid(span, span, indexing="ij")
        goal_grid = jnp.asarray(np.stack([gx.ravel(), gy.ravel()], -1))
        n = goal_grid.shape[0]
        sb = jnp.repeat(s, n, 0)
        zero_a = jnp.zeros((n, ACTION_SIZE))

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
        for ax, (name, q) in zip(axes, [
            ("CRL", crl_q(sb, goal_grid, zero_a)),
            (f"CARL-CRL (n={carl_cfg.action_seq_len})",
             carl_q(sb, goal_grid, jnp.tile(zero_a, (1, carl_cfg.action_seq_len)))),
        ]):
            q = np.asarray(q).reshape(span.size, -1)
            im = ax.pcolormesh(span, span, q.T, shading="auto", cmap="magma")
            ax.plot(np.asarray(s)[0, GOAL_IDX[0]], np.asarray(s)[0, GOAL_IDX[1]],
                    "co", ms=9, mec="white", label="agent")
            ax.set_title(f"{name}\nQ range [{q.min():.2f}, {q.max():.2f}]")
            ax.set_xlabel("goal x"); ax.set_ylabel("goal y"); ax.legend(loc="upper right")
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"Q over goal position, state {which.value}, action = 0")
        return fig

    _figure()
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Do they rank actions the same way?

        The two critics sit on different scales (and `energy_fn = norm` makes both negative), so
        absolute values are not comparable. What is comparable is the *ordering* they impose over
        actions -- that is what the actor follows.
        """
    )
    return


@app.cell
def _(ACTION_SIZE, carl_cfg, carl_q, crl_q, goals, jax, jnp, np, plt, states):
    def _figure():
        n_samp = 512
        rows = []
        for i in range(states.shape[0]):
            a = jax.random.uniform(jax.random.PRNGKey(i), (n_samp, ACTION_SIZE), minval=-1, maxval=1)
            sb = jnp.repeat(states[i:i + 1], n_samp, 0)
            gb = jnp.repeat(goals[i:i + 1], n_samp, 0)
            qc = np.asarray(crl_q(sb, gb, a))
            qk = np.asarray(carl_q(sb, gb, jnp.tile(a, (1, carl_cfg.action_seq_len))))
            rank = np.corrcoef(np.argsort(np.argsort(qc)), np.argsort(np.argsort(qk)))[0, 1]
            rows.append((i, rank, qc, qk))

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
        axes[0].bar([r[0] for r in rows], [r[1] for r in rows])
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].set_xlabel("fixed state"); axes[0].set_ylabel("Spearman rank corr")
        axes[0].set_title("agreement on the ordering of 512 random actions")
        axes[1].scatter(rows[0][2], rows[0][3], s=6, alpha=0.4)
        axes[1].set_xlabel("CRL Q"); axes[1].set_ylabel("CARL-CRL Q")
        axes[1].set_title(f"state 0, rank corr = {rows[0][1]:.3f}")
        return fig

    _figure()
    return
