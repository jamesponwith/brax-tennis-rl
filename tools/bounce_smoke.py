"""Bounce-fidelity smoke test: does ball restitution behave per backend?

Drops a tennis-ball-sized sphere from 1m onto a plane in each candidate
backend and reports the peak height after the first bounce, plus sanity
flags (NaN, floor penetration). One-shot decision tooling for the backend
choice — the result and the pick live in docs/adr/0002-backend-choice.md.

Run: uv run python tools/bounce_smoke.py
"""

import importlib
import math

import jax
import jax.numpy as jp
from brax.io import mjcf

# underdamped contact → restitution; ball ≈ tennis ball. dampratio 0.15 is
# the chosen operating point (ADR 0002): rebound ≈ 33% (CoR ≈ 0.57), the
# most bounce MJX gives before the solver starts injecting energy (< 0.15
# the 1 m drop rebounds > 1 m). ponytail: smaller timestep may extend the
# stable range if Phase 2 needs a livelier ball.
MJCF = """
<mujoco>
  <option timestep="0.004"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" solref="0.02 0.15"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.033" mass="0.057" solref="0.02 0.15"/>
    </body>
  </worldbody>
</mujoco>
"""

BALL_R = 0.033
SECONDS = 2.0


def simulate(backend: str) -> list[float]:
    sys = mjcf.loads(MJCF)
    pipeline = importlib.import_module(f"brax.{backend}.pipeline")
    state = jax.jit(pipeline.init)(sys, sys.init_q, jp.zeros(sys.qd_size()))
    step = jax.jit(pipeline.step)
    act = jp.zeros(sys.act_size())
    heights = []
    for _ in range(int(SECONDS / sys.opt.timestep)):
        state = step(sys, state, act)
        heights.append(float(state.x.pos[0, 2]))
    return heights


def report(backend: str) -> None:
    try:
        h = simulate(backend)
    except Exception as e:  # noqa: BLE001 — any failure to sim disqualifies the backend
        print(f"{backend:12s} FAILED: {e}")
        return
    if any(math.isnan(x) for x in h):
        print(f"{backend:12s} NaN in trajectory")
        return
    floor_pen = min(h) < BALL_R * 0.5
    # first touchdown, then the rebound peak after it
    touchdown = next((i for i, x in enumerate(h) if x <= BALL_R * 1.1), None)
    if touchdown is None:
        print(f"{backend:12s} never reached the floor?!")
        return
    rebound = max(h[touchdown:])
    print(
        f"{backend:12s} rebound peak {rebound:.3f} m from 1.0 m drop "
        f"({rebound:.0%}), min height {min(h):.3f} m"
        f"{'  ⚠ floor penetration' if floor_pen else ''}"
    )


if __name__ == "__main__":
    for backend in ("mjx", "positional"):
        report(backend)
