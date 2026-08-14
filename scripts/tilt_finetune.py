"""Tilt curriculum: warm-start Phase 2's flat-face champion with the tilt
hinge enabled. Training WITH tilt from scratch never leaves the random
baseline (see docs/writeup-phase2.md); starting from a policy that already
plays, with zero-initialized tilt pathways, the wrist gets learned instead
of flailed.

  uv run python scripts/tilt_finetune.py out/phase2_params
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jp
from brax.io import model
from brax.training.agents.ppo import train as ppo

from envs.tennis import PHASE2_CONFIG, Tennis

TILT_CONFIG = dataclasses.replace(PHASE2_CONFIG, orientation=True)


def expand_params(params, obs_add: int = 2):
    """Grow (normalizer, policy, value) from obs 10/act 2 to obs 12/act 3.

    New obs rows and the tilt action column start at zero: identical behavior
    on step one, gradients decide what the wrist is for.
    """
    norm, policy, value = params

    def grow_obs(arr):  # (10,) stats -> (12,)
        pad = jp.zeros(obs_add) if arr.ndim == 1 else None
        return jp.concatenate([arr, pad]) if pad is not None else arr

    # new dims get std=1: summed_variance must scale with the running count
    norm = norm.replace(
        mean=grow_obs(norm.mean),
        std=jp.concatenate([norm.std, jp.ones(obs_add)]),
        summed_variance=jp.concatenate([norm.summed_variance, jp.full(obs_add, norm.count)]),
    )

    def grow_input(net):
        k = net["params"]["hidden_0"]["kernel"]  # (10, width)
        net["params"]["hidden_0"]["kernel"] = jp.concatenate(
            [k, jp.zeros((obs_add, k.shape[1]))], axis=0
        )
        return net

    policy = grow_input(policy)
    value = grow_input(value)

    # policy head: NormalTanh params are [means | log_stds]; act 2 -> 3 means
    # inserting a zero column after each half's old columns
    last = max(policy["params"].keys())
    k = policy["params"][last]["kernel"]  # (width, 4)
    b = policy["params"][last]["bias"]  # (4,)
    zk = jp.zeros((k.shape[0], 1))
    policy["params"][last]["kernel"] = jp.concatenate([k[:, :2], zk, k[:, 2:], zk], axis=1)
    # tilt log-std bias -4: softplus(-4) ≈ 0.02 → a QUIET wrist. Bias 0 gives
    # std ~0.7 — full random tilt from step one, the exact face-flattening
    # poison bisection found; run 1 of this script reproduced it (0.2%
    # returned from a 71.7% warm start).
    policy["params"][last]["bias"] = jp.concatenate([b[:2], jp.zeros(1), b[2:], jp.full(1, -4.0)])
    return (norm, policy, value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("params", type=Path)
    ap.add_argument("--timesteps", type=int, default=6_000_000)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    restore = expand_params(model.load_params(args.params))
    env = Tennis(TILT_CONFIG)
    xs, rates = [], []

    def progress(steps, m):
        r = float(m["eval/episode_returned"])
        xs.append(steps)
        rates.append(r)
        print(
            f"{steps:>11,} steps  returned {r:6.1%}  "
            f"contact {float(m['eval/episode_interception']):6.1%}  "
            f"reward {float(m['eval/episode_reward']):8.2f}",
            flush=True,
        )

    _, params, _ = ppo.train(
        environment=env,
        progress_fn=progress,
        seed=7,
        restore_params=restore,
        num_timesteps=args.timesteps,
        num_evals=12,
        num_eval_envs=1000,
        reward_scaling=1.0,
        episode_length=200,
        normalize_observations=True,
        unroll_length=10,
        num_minibatches=16,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=1e-4,  # fine-tune, don't stampede
        entropy_cost=5e-3,
        num_envs=1024,
        batch_size=512,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_params(args.out / "phase2_tilt_params", params)
    print(f"final returned with tilt: {rates[-1]:.1%} (flat baseline 71.7%)")


if __name__ == "__main__":
    main()
