"""Phase 1 training: PPO on the tennis interception env.

The success metric rides brax's own evaluator: `interception` is 1.0 only on
the contact step, so eval/episode_interception == interception rate over
num_eval_envs episodes. Target: >= 0.95 over 1000 serves (SPEC).

  uv run python scripts/train_tennis.py                 # full run
  uv run python scripts/train_tennis.py --smoke         # CPU mechanics check
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root for envs/

from brax.io import html, model
from brax.training.agents.ppo import train as ppo

from envs import Tennis

FULL = {
    "num_timesteps": 8_000_000,
    "num_evals": 20,
    "num_eval_envs": 1000,  # SPEC: rate measured over 1000 serves
    "reward_scaling": 1.0,
    "episode_length": 200,  # 4 s at dt=0.02 — serves arrive well within this
    "normalize_observations": True,
    "unroll_length": 10,
    "num_minibatches": 16,
    "num_updates_per_batch": 4,
    "discounting": 0.97,
    "learning_rate": 3e-4,
    "entropy_cost": 1e-2,
    "num_envs": 1024,
    "batch_size": 512,
}
SMOKE = {
    "num_timesteps": 100_000,
    "num_evals": 2,
    "num_eval_envs": 128,
    "num_envs": 128,
    "batch_size": 128,
    "num_minibatches": 4,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    hp = FULL | (SMOKE if args.smoke else {})
    env = Tennis()
    xs, rates = [], []

    def progress(num_steps: int, metrics: dict) -> None:
        rate = float(metrics["eval/episode_interception"])
        xs.append(num_steps)
        rates.append(rate)
        print(
            f"{num_steps:>11,} steps  interception {rate:6.1%}  "
            f"reward {float(metrics['eval/episode_reward']):8.2f}",
            flush=True,
        )

    make_inference_fn, params, _ = ppo.train(
        environment=env, progress_fn=progress, seed=args.seed, **hp
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_params(args.out / "tennis_params", params)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.plot(xs, rates, marker="o")
    plt.axhline(0.95, ls="--", c="r", label="target 95%")
    plt.xlabel("env steps")
    plt.ylabel(f"interception rate ({hp['num_eval_envs']} serves)")
    plt.title("tennis interception — PPO")
    plt.legend()
    plt.savefig(args.out / "phase1_curve.png", dpi=120, bbox_inches="tight")

    _render_rollout(env, make_inference_fn(params), args.out / "phase1_rollout.html")
    print(f"final interception rate: {rates[-1]:.1%}  (artifacts in {args.out}/)")


def _render_rollout(env: Tennis, inference_fn, out: Path, episodes: int = 5) -> None:
    import jax

    jit_reset, jit_step, jit_infer = jax.jit(env.reset), jax.jit(env.step), jax.jit(inference_fn)
    rng = jax.random.PRNGKey(0)
    frames = []
    for _ in range(episodes):
        rng, r = jax.random.split(rng)
        state = jit_reset(r)
        for _ in range(200):
            frames.append(state.pipeline_state)
            rng, r = jax.random.split(rng)
            act, _ = jit_infer(state.obs, r)
            state = jit_step(state, act)
            if state.done:
                break
    out.write_text(html.render(env.sys.tree_replace({"opt.timestep": env.dt}), frames))
    print(f"rendered {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
