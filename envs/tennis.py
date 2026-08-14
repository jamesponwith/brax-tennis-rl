"""Tennis interception env — Phase 1 (SPEC).

The paddle IS the player: an actuated box sliding in x/y at the baseline.
A ball is served from the far half within a randomized cone; the agent's job
is to be where the ball arrives.

Observation (10): ball pos (3), ball vel (3), paddle pos (2), paddle vel (2).
Action (2): paddle target velocity in x/y, [-1, 1] scaled by max_paddle_speed.
Reward: -distance from paddle to the ball's predicted landing point each step
(dense shaping), +contact_bonus on interception. Per-term components live in
`state.metrics` from day one so reward hacks show up in curves (SPEC risk).
Episode ends on contact or ball-past-paddle; timeout is the trainer's
episode_length truncation.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

G = 9.81
BALL_R = 0.033


@dataclass(frozen=True)
class TennisConfig:
    """Phase 1 knobs. Net / target region / paddle orientation arrive in Phase 2."""

    court_half_w: float = 4.0  # lateral serve spread (m)
    paddle_y: float = -6.0  # baseline the paddle lives on
    serve_y: float = 8.0  # where serves spawn
    serve_h_lo: float = 1.0
    serve_h_hi: float = 2.5
    serve_speed_lo: float = 8.0
    serve_speed_hi: float = 13.0
    serve_cone_deg: float = 12.0  # lateral cone half-angle around straight-at-paddle
    serve_vz_lo: float = -1.0
    serve_vz_hi: float = 2.0
    max_paddle_speed: float = 8.0  # m/s at |action| = 1
    contact_bonus: float = 10.0
    contact_dist: float = 0.6  # ponytail: proximity check, not MJX contact parsing
    shaping_scale: float = 0.1  # per-step weight on -distance-to-landing


# solref 0.02/0.15 per ADR 0002 — dampratio < 0.15 injects energy.
_MJCF = """
<mujoco>
  <option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="court" type="plane" size="20 20 0.1" solref="0.02 0.15"/>
    <body name="paddle" pos="0 {paddle_y} 0.5">
      <joint name="px" type="slide" axis="1 0 0" damping="2"/>
      <joint name="py" type="slide" axis="0 1 0" damping="2"/>
      <geom name="paddle" type="box" size="0.5 0.15 0.5" mass="1"/>
    </body>
    <body name="ball" pos="0 {serve_y} 1.5">
      <freejoint/>
      <geom name="ball" type="sphere" size="{ball_r}" mass="0.057" solref="0.02 0.15"/>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="px" kv="20" ctrlrange="-10 10"/>
    <velocity joint="py" kv="20" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def predict_landing(pos: jp.ndarray, vel: jp.ndarray) -> jp.ndarray:
    """xy where the ball's current ballistic arc reaches ball height.

    ponytail: ignores future bounces — after a bounce the arc re-predicts
    next step anyway, so the shaping keeps pointing along the ball's line.
    """
    z = jp.maximum(pos[2] - BALL_R, 0.0)
    t = (vel[2] + jp.sqrt(vel[2] ** 2 + 2.0 * G * z)) / G
    return pos[:2] + vel[:2] * t


class Tennis(PipelineEnv):
    def __init__(self, config: TennisConfig | None = None):
        self.cfg = config = config or TennisConfig()
        sys = mjcf.loads(
            _MJCF.format(paddle_y=config.paddle_y, serve_y=config.serve_y, ball_r=BALL_R)
        )
        super().__init__(sys=sys, backend="mjx", n_frames=5)  # dt = 0.02

    def reset(self, rng: jax.Array) -> State:
        cfg = self.cfg
        r_x, r_h, r_speed, r_ang, r_vz = jax.random.split(rng, 5)
        x0 = jax.random.uniform(r_x, minval=-cfg.court_half_w, maxval=cfg.court_half_w)
        h = jax.random.uniform(r_h, minval=cfg.serve_h_lo, maxval=cfg.serve_h_hi)
        speed = jax.random.uniform(r_speed, minval=cfg.serve_speed_lo, maxval=cfg.serve_speed_hi)
        cone = jp.deg2rad(cfg.serve_cone_deg)
        ang = jax.random.uniform(r_ang, minval=-cone, maxval=cone)
        vz = jax.random.uniform(r_vz, minval=cfg.serve_vz_lo, maxval=cfg.serve_vz_hi)

        # q: [paddle_x, paddle_y_offset, ball_pos(3), ball_quat(4)]
        q = jp.concatenate(
            [jp.zeros(2), jp.array([x0, cfg.serve_y, h]), jp.array([1.0, 0.0, 0.0, 0.0])]
        )
        # qd: [paddle(2), ball_lin(3), ball_ang(3)]; serve flies toward the paddle (-y)
        vxy = speed * jp.array([jp.sin(ang), -jp.cos(ang)])
        qd = jp.concatenate([jp.zeros(2), vxy, jp.array([vz]), jp.zeros(3)])

        ps = self.pipeline_init(q, qd)
        metrics = {"reward_shaping": 0.0, "reward_contact": 0.0, "interception": 0.0}
        return State(ps, self._obs(ps), jp.float32(0.0), jp.float32(0.0), metrics)

    def step(self, state: State, action: jax.Array) -> State:
        cfg = self.cfg
        ctrl = jp.clip(action, -1.0, 1.0) * cfg.max_paddle_speed
        ps = self.pipeline_step(state.pipeline_state, ctrl)

        ball_pos, ball_vel = ps.x.pos[1], ps.xd.vel[1]
        paddle_pos = ps.x.pos[0]

        landing = predict_landing(ball_pos, ball_vel)
        shaping = -cfg.shaping_scale * jp.linalg.norm(paddle_pos[:2] - landing)
        contact = jp.linalg.norm(ball_pos - paddle_pos) < cfg.contact_dist
        past = ball_pos[1] < cfg.paddle_y - 0.5

        reward_contact = jp.where(contact, cfg.contact_bonus, 0.0)
        reward = shaping + reward_contact
        done = jp.float32(contact | past)

        # in-place update keeps keys the training wrappers add to metrics
        state.metrics.update(
            reward_shaping=shaping,
            reward_contact=reward_contact,
            interception=jp.float32(contact),
        )
        return state.replace(
            pipeline_state=ps, obs=self._obs(ps), reward=jp.float32(reward), done=done
        )

    def _obs(self, ps) -> jp.ndarray:
        # ball pos(3) + vel(3), paddle xy pos(2) + xy vel(2)
        return jp.concatenate([ps.x.pos[1], ps.xd.vel[1], ps.x.pos[0][:2], ps.xd.vel[0][:2]])
