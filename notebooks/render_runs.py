"""Render the final policy of any finished run, without re-training it.

The sweep only rendered at the first eval (`--visualization_interval 999`), so wandb has a single
early rollout per run and no slider. Every run does save its final parameters to
`runs/run_<exp_name>_s_<seed>/ckpt/final`, though, so the interesting rollout -- the trained
policy -- can be reconstructed here on demand.

Run it with:  uv run --with marimo marimo edit notebooks/render_runs.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # Small rollouts; keep off the GPU so a running sweep is undisturbed.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import pickle
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
    from brax.io import html, model
    from brax.training.acme import running_statistics

    from jaxgcrl.utils.env import create_env

    return (
        Path,
        create_env,
        html,
        jax,
        jnp,
        mo,
        model,
        pickle,
        running_statistics,
    )


@app.cell
def _(Path, mo, pickle):
    def load_cfg(run_dir):
        with open(run_dir / "args.pkl", "rb") as f:
            cfg = pickle.load(f)
        return cfg["agent"], cfg["run"]

    # Scoped to seed 1 of the carl-chunking sweep, whose exp_names look like "<agent>__<env>__s1".
    # Widen by relaxing this filter -- every run.py invocation leaves the same two files behind.
    RUNS = sorted(p for p in Path("runs").glob("run_*__*__s1_s_1")
                  if (p / "ckpt" / "final").exists() and (p / "args.pkl").exists())
    options = {p.name.replace("run_", "").replace("_s_1", ""): p for p in RUNS}

    pick = mo.ui.dropdown(options=options, label="run",
                          value=next(iter(options)) if options else None)
    steps = mo.ui.slider(200, 2000, value=1000, step=200, label="steps", show_value=True)
    seed = mo.ui.number(0, 20, value=1, label="seed")
    go = mo.ui.run_button(label="render")
    save = mo.ui.run_button(label="save html")

    # One control block, so the buttons sit next to what they produce.
    mo.vstack([mo.md(f"**{len(RUNS)}** runs with a final checkpoint."),
               mo.hstack([pick, steps, seed, go, save], justify="start", gap=1, wrap=True)])
    return go, load_cfg, pick, save, seed, steps


@app.cell
def _(jnp, load_cfg, model, running_statistics):
    def policy_for(run_dir, env):
        """A deterministic `policy(obs_batch, key) -> (action, {})` for whichever agent ran here.

        Every agent saves a different tuple, and two of them save something `make_policy` cannot
        consume directly (crl and carl_crl save the critic alongside the actor), so each family is
        rebuilt from its own modules rather than through the training-time closure.
        """
        agent, _ = load_cfg(run_dir)
        saved = model.load_params(str(run_dir / "ckpt" / "final"))
        kind = type(agent).__name__
        S, A = env.state_dim, env.action_size
        obs_size = S + len(env.goal_indices)
        # the baseline TD3 config carries no width/depth at all -- it takes brax's defaults
        trunk = dict(network_width=getattr(agent, "h_dim", 256),
                     network_depth=getattr(agent, "n_hidden", 2),
                     skip_connections=getattr(agent, "skip_connections", 0),
                     use_relu=getattr(agent, "use_relu", False),
                     use_ln=getattr(agent, "use_ln", False))

        if kind == "CRL":
            from jaxgcrl.agents.crl.networks import Actor
            _, actor_params, _ = saved
            actor = Actor(action_size=A, **trunk)

            def policy(obs, key):
                mean, _ = actor.apply(actor_params, obs)
                return jnp.tanh(mean), {}          # the training-time renderer skips this tanh

        elif kind == "CarlCRL":
            from jaxgcrl.agents.carl_crl.networks import Actor, Encoder, sample_actions
            _, actor_params, critic = saved
            actor = Actor(action_size=A, action_seq_len=agent.action_seq_len, **trunk)
            sg = Encoder(repr_dim=agent.repr_dim, **trunk)

            def policy(obs, key):
                chunk, _ = sample_actions(
                    {"actor": actor, "sg_encoder": sg},
                    {"actor": actor_params, "sg_encoder": critic["sg_encoder"]},
                    obs[:, :S], obs[:, S:], agent.actor_conditioning)
                return chunk[..., :A], {}          # execute the first action, replan next step

        elif kind in ("CarlSAC", "CarlTD3"):
            mod = (__import__("jaxgcrl.agents.carl_sac.networks", fromlist=["x"])
                   if kind == "CarlSAC"
                   else __import__("jaxgcrl.agents.carl_td3.networks", fromlist=["x"]))
            ctx = dict(**vars(agent), obs_size=obs_size, state_size=S, action_size=A)
            nets = mod.make_networks(ctx)
            pre = (running_statistics.normalize if agent.normalize_observations
                   else (lambda x, y: x))
            policy = mod.make_policy(ctx, nets, pre, saved, deterministic=True)

        elif kind in ("SAC", "TD3"):
            mod = (__import__("jaxgcrl.agents.sac.networks", fromlist=["x"]) if kind == "SAC"
                   else __import__("jaxgcrl.agents.td3.networks", fromlist=["x"]))
            pre = (running_statistics.normalize if agent.normalize_observations
                   else (lambda x, y: x))
            kw = dict(preprocess_observations_fn=pre)
            if kind == "SAC":
                kw.update(hidden_layer_sizes=[agent.h_dim] * agent.n_hidden, layer_norm=agent.use_ln)
                nets = mod.make_sac_networks(obs_size, A, **kw)
                policy = mod.make_inference_fn(nets)(saved, deterministic=True)
            else:
                nets = mod.make_td3_networks(obs_size, A, **kw)
                policy = mod.make_inference_fn(nets)(saved, exploration_noise=0, noise_clip=0,
                                                     deterministic=True)
        else:
            raise ValueError(f"no policy adapter for agent {kind}")

        return policy, agent, kind

    return (policy_for,)


@app.cell
def _(html, jax):
    def rollout_html(policy, env, steps, seed, reset_every=1000, height=600):
        """Roll the policy out and return brax's viewer as an HTML string.

        This is `jaxgcrl.utils.env.render` minus the `wandb.log` at the end, so it can run with no
        active wandb run.
        """
        jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
        jit_policy = jax.jit(policy)

        key = jax.random.PRNGKey(seed)
        key, sub = jax.random.split(key)
        state = jit_reset(rng=sub)
        frames = []
        for i in range(steps):
            frames.append(state.pipeline_state)
            key, sub = jax.random.split(key)
            action, _ = jit_policy(state.obs[None], sub)
            state = jit_step(state, action[0])
            if (i + 1) % reset_every == 0:
                key, sub = jax.random.split(key)
                state = jit_reset(rng=sub)
        return html.render(env.sys.tree_replace({"opt.timestep": env.dt}), frames, height=height)

    return (rollout_html,)


@app.cell
def _(
    create_env,
    go,
    load_cfg,
    mo,
    pick,
    policy_for,
    rollout_html,
    seed,
    steps,
):
    mo.stop(not go.value, mo.md("Pick a run and press **render**."))

    agent_cfg, run_cfg = load_cfg(pick.value)
    env = create_env(run_cfg.env)
    policy, agent, kind = policy_for(pick.value, env)
    page = rollout_html(policy, env, steps.value, int(seed.value))

    mo.vstack([
        mo.md(f"**{pick.value.name}** — {kind} on `{run_cfg.env}`, "
              f"{steps.value} steps, seed {seed.value}"),
        mo.iframe(page, height=620),
    ])
    return (page,)


@app.cell
def _(mo, page, pick, save):
    mo.stop(not save.value, mo.md(""))
    out = pick.value / f"{pick.value.name}_final.html"
    out.write_text(page)
    mo.md(f"wrote `{out}`")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Getting these into wandb

    `wandb.init(project="jaxgcrl", id=<run_id>, resume="must")` then
    `wandb.log({"render": wandb.Html(page)})` appends to an existing run — the run id is the
    suffix of the local `wandb/run-<timestamp>-<id>` directory, or `wandb.Api()` can look it
    up by name. Logging the `render` key at several steps is what produces the slider.

    For future sweeps the cheaper fix is `--visualization_interval 7` with `--num_evals 50`:
    `do_render` is `ne % interval == 0`, so that renders at evals 0, 7, ... 49 — eight
    rollouts including the final policy, rather than only the first.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
