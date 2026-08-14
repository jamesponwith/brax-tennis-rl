"""Phase 3 training: round-based self-play.

Each round trains the learner against a frozen opponent, then the opponent
becomes the freshly trained policy for the next round (league of one — the
latest; a sampled league of past checkpoints is the upgrade if cycling shows
up in the rally-length curve). Both sides start from the Phase 2 champion.

  uv run python scripts/train_rally.py out/phase2_params --rounds 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from envs.rally import RallyTennis

HP = {
    "num_evals": 6,
    "num_eval_envs": 1000,
    "reward_scaling": 1.0,
    "episode_length": 400,  # rallies outlive single points
    "normalize_observations": True,
    "unroll_length": 10,
    "num_minibatches": 16,
    "num_updates_per_batch": 4,
    "discounting": 0.97,
    "learning_rate": 1e-4,
    "entropy_cost": 5e-3,
    "num_envs": 1024,
    "batch_size": 512,
}


def opponent_fn(params):
    nets = ppo_networks.make_ppo_networks(
        10, 2, preprocess_observations_fn=running_statistics.normalize
    )
    policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    rng = jax.random.PRNGKey(0)  # unused in deterministic mode
    return lambda obs: policy(obs, rng)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("params", type=Path)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps-per-round", type=int, default=3_000_000)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    current = model.load_params(args.params)
    xs, rallies = [], []
    offset = 0

    for rnd in range(args.rounds):
        env = RallyTennis()
        env.set_opponent(opponent_fn(current))

        def progress(steps, m, _offset=offset, _rnd=rnd):
            r = float(m["eval/episode_rally_length"])
            xs.append(_offset + steps)
            rallies.append(r)
            print(
                f"round {_rnd}  {steps:>10,} steps  rally {r:5.2f}  "
                f"returned {float(m['eval/episode_returned']):5.2f}  "
                f"reward {float(m['eval/episode_reward']):8.2f}",
                flush=True,
            )

        _, current, _ = ppo.train(
            environment=env,
            progress_fn=progress,
            seed=100 + rnd,
            restore_params=current,
            num_timesteps=args.steps_per_round,
            **HP,
        )
        offset += args.steps_per_round
        model.save_params(args.out / f"rally_r{rnd}_params", current)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_params(args.out / "rally_params", current)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.plot(xs, rallies, marker="o")
    plt.xlabel("env steps (across rounds)")
    plt.ylabel("mean rally length (net crossings, 1000 episodes)")
    plt.title("rally self-play — PPO league")
    plt.savefig(args.out / "phase3_curve.png", dpi=120, bbox_inches="tight")
    print(f"final mean rally length: {rallies[-1]:.2f} (probe baseline 1.80)")


if __name__ == "__main__":
    main()
