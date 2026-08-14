"""Re-render demo rollouts from a saved checkpoint — deterministic best play
on the current (colorized) env, without retraining.

  uv run python scripts/render_tennis.py out/phase1_params
  uv run python scripts/render_tennis.py out/phase2_params --phase2 --serves 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root for envs/

import jax
from brax.io import html, model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from envs.tennis import Tennis, TennisConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("params", type=Path)
    ap.add_argument("--phase2", action="store_true")
    ap.add_argument("--serves", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = (
        Tennis(TennisConfig(net=True, serve_h_lo=1.8, serve_h_hi=3.0, paddle_half_h=1.0))
        if args.phase2
        else Tennis()
    )
    # mirror ppo.train's default network factory so saved params fit
    nets = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    inference = ppo_networks.make_inference_fn(nets)(
        model.load_params(args.params), deterministic=True
    )

    jit_reset, jit_step, jit_infer = jax.jit(env.reset), jax.jit(env.step), jax.jit(inference)
    rng = jax.random.PRNGKey(args.seed)
    frames, wins = [], 0
    metric = "returned" if args.phase2 else "interception"
    for i in range(args.serves):
        rng, r = jax.random.split(rng)
        state = jit_reset(r)
        for _ in range(300):
            frames.append(state.pipeline_state)
            rng, r = jax.random.split(rng)
            act, _ = jit_infer(state.obs, r)
            state = jit_step(state, act)
            if state.done:
                break
        won = bool(state.metrics[metric] == 1.0)
        wins += won
        print(f"serve {i + 1}: {'✓ ' + metric if won else '✗'}")

    out = args.out or args.params.parent / f"{args.params.name.replace('_params', '')}_rollout.html"
    out.write_text(html.render(env.sys.tree_replace({"opt.timestep": env.dt}), frames))
    print(f"{wins}/{args.serves} {metric}  ->  {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
