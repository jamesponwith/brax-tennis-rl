"""Tennis env — Phase 1 (interception) and Phase 2 (the return), by config.

The paddle IS the player: an actuated box sliding in x/y at the baseline,
optionally with a tilt hinge (Phase 2) for flat-through-lob control.

Phase 1 (net=False): intercept the serve. Obs (10): ball pos(3) + vel(3),
paddle pos(2) + vel(2). Action (2): target velocity. Reward: -distance from
paddle to the ball's predicted landing point each step, +bonus on contact.
Episode ends on contact or ball-past-paddle.

Phase 2 (net=True, orientation=True): return the serve over the net into a
target region. Obs (12): + paddle tilt angle and angular velocity. Action
(3): + tilt target velocity. The episode continues past contact; return
detection is MEMORYLESS — serves always fly -y, so ball_vy > 0 means "the
paddle sent it back" (brax's AutoResetWrapper leaks custom info flags across
resets, so episode memory is a trap). A netted ball comes back with vy < 0
and is simply in play again. Reward: Phase 1 shaping while incoming, +contact
bonus on the vy sign flip at the paddle, +return bonus when the ball lands
beyond the net, +target bonus scaled by proximity to the target center.

Per-term reward components live in `state.metrics` from day one so reward
hacks show up in curves (SPEC risk). Metrics must be updated in place —
training wrappers add their own keys.
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
    paddle_half_h: float = 0.5  # Phase 2 raises this: lobbed serves hop a short paddle
    shaping_scale: float = 0.1  # per-step weight on -distance-to-landing
    # Phase 2
    net: bool = False
    orientation: bool = False
    net_height: float = 0.9
    return_bonus: float = 10.0  # ball lands beyond the net
    target_bonus: float = 10.0  # scaled by proximity to (0, target_y)
    target_y: float = 5.0
    target_radius: float = 3.0


# solref 0.02/0.15 per ADR 0002 — dampratio < 0.15 injects energy.
_MJCF = """
<mujoco>
  <option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="court" type="plane" size="20 20 0.1" solref="0.02 0.15" rgba="0.16 0.42 0.25 1"/>
    <geom name="line_near" type="box" pos="0 {paddle_y} 0.005" size="5 0.04 0.005"
          contype="0" conaffinity="0" rgba="1 1 1 0.9"/>
    <geom name="line_far" type="box" pos="0 6 0.005" size="5 0.04 0.005"
          contype="0" conaffinity="0" rgba="1 1 1 0.9"/>
    <geom name="line_left" type="box" pos="-5 0 0.005" size="0.04 6 0.005"
          contype="0" conaffinity="0" rgba="1 1 1 0.9"/>
    <geom name="line_right" type="box" pos="5 0 0.005" size="0.04 6 0.005"
          contype="0" conaffinity="0" rgba="1 1 1 0.9"/>
    {net}
    <body name="paddle" pos="0 {paddle_y} {paddle_half_h}">
      <joint name="px" type="slide" axis="1 0 0" damping="2"/>
      <joint name="py" type="slide" axis="0 1 0" damping="2" range="-2 2"/>
      {tilt_joint}
      <geom name="paddle" type="box" size="0.5 0.15 {paddle_half_h}" mass="1" solref="0.02 0.15" rgba="0.16 0.32 0.75 1"/>
    </body>
    <body name="ball" pos="0 {serve_y} 1.5">
      <freejoint/>
      <geom name="ball" type="sphere" size="{ball_r}" mass="0.057" solref="0.02 0.15" rgba="0.85 0.93 0.16 1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="px" kv="20" ctrlrange="-10 10"/>
    <velocity joint="py" kv="20" ctrlrange="-10 10"/>
    {tilt_act}
  </actuator>
</mujoco>
"""
_NET = """<geom name="net" type="box" pos="0 0 {h2}" size="6 0.02 {h2}" solref="0.02 0.5" rgba="0.92 0.94 0.96 0.85"/>
    <geom name="post_l" type="box" pos="-6 0 {h2}" size="0.05 0.05 {h2}" contype="0" conaffinity="0" rgba="0.25 0.25 0.28 1"/>
    <geom name="post_r" type="box" pos="6 0 {h2}" size="0.05 0.05 {h2}" contype="0" conaffinity="0" rgba="0.25 0.25 0.28 1"/>
    <geom name="target_patch" type="box" pos="0 {ty} 0.004" size="1 1 0.003" contype="0" conaffinity="0" rgba="1 1 1 0.16"/>"""
_TILT_JOINT = (
    '<joint name="ptilt" type="hinge" axis="1 0 0" range="-60 60" damping="0.5" stiffness="25"/>'
)
_TILT_ACT = '<velocity joint="ptilt" kv="5" ctrlrange="-6 6"/>'


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
        xml = _MJCF.format(
            paddle_y=config.paddle_y,
            paddle_half_h=config.paddle_half_h,
            serve_y=config.serve_y,
            ball_r=BALL_R,
            net=_NET.format(h2=config.net_height / 2, ty=config.target_y) if config.net else "",
            tilt_joint=_TILT_JOINT if config.orientation else "",
            tilt_act=_TILT_ACT if config.orientation else "",
        )
        super().__init__(sys=mjcf.loads(xml), backend="mjx", n_frames=5)  # dt = 0.02
        self._npad = 3 if config.orientation else 2  # paddle joint count

    def reset(self, rng: jax.Array) -> State:
        cfg = self.cfg
        r_x, r_h, r_speed, r_ang, r_vz = jax.random.split(rng, 5)
        x0 = jax.random.uniform(r_x, minval=-cfg.court_half_w, maxval=cfg.court_half_w)
        h = jax.random.uniform(r_h, minval=cfg.serve_h_lo, maxval=cfg.serve_h_hi)
        speed = jax.random.uniform(r_speed, minval=cfg.serve_speed_lo, maxval=cfg.serve_speed_hi)
        cone = jp.deg2rad(cfg.serve_cone_deg)
        ang = jax.random.uniform(r_ang, minval=-cone, maxval=cone)
        vz = jax.random.uniform(r_vz, minval=cfg.serve_vz_lo, maxval=cfg.serve_vz_hi)

        q = jp.concatenate(
            [jp.zeros(self._npad), jp.array([x0, cfg.serve_y, h]), jp.array([1.0, 0.0, 0.0, 0.0])]
        )
        vxy = speed * jp.array([jp.sin(ang), -jp.cos(ang)])
        if cfg.net:
            # a serve must clear the net or it rebounds with vy > 0 and the
            # memoryless return detector counts a phantom return — caught by
            # the per-term metrics on eval #1 (random policy "returned" 85%)
            t_net = cfg.serve_y / (speed * jp.cos(ang))
            vz_min = (cfg.net_height + 0.15 - h) / t_net + 0.5 * G * t_net
            vz = jp.maximum(vz, vz_min)
        qd = jp.concatenate([jp.zeros(self._npad), vxy, jp.array([vz]), jp.zeros(3)])

        ps = self.pipeline_init(q, qd)
        metrics = {"reward_shaping": 0.0, "reward_contact": 0.0, "interception": 0.0}
        if cfg.net:
            metrics |= {"reward_return": 0.0, "reward_target": 0.0, "returned": 0.0}
        return State(ps, self._obs(ps), jp.float32(0.0), jp.float32(0.0), metrics)

    def step(self, state: State, action: jax.Array) -> State:
        cfg = self.cfg
        prev_vy = state.obs[4]
        # slide channels in m/s, tilt channel (if any) in rad/s
        scale = (
            jp.array([cfg.max_paddle_speed, cfg.max_paddle_speed, 6.0])
            if cfg.orientation
            else cfg.max_paddle_speed
        )
        ctrl = jp.clip(action, -1.0, 1.0) * scale
        ps = self.pipeline_step(state.pipeline_state, ctrl)

        ball_pos, ball_vel = ps.x.pos[1], ps.xd.vel[1]
        paddle_pos = ps.x.pos[0]

        incoming = ball_vel[1] <= 0.0  # serves fly -y; vy > 0 only after a paddle hit
        landing = predict_landing(ball_pos, ball_vel)
        shaping = jp.where(
            incoming, -cfg.shaping_scale * jp.linalg.norm(paddle_pos[:2] - landing), 0.0
        )
        # box-aware proximity: a tall paddle face means center distance misleads
        d = jp.abs(ball_pos - paddle_pos)
        near = (
            (d[0] < 0.5 + 2 * BALL_R)
            & (d[1] < 0.15 + 2 * BALL_R + 0.1)
            & (ball_pos[2] <= 2 * cfg.paddle_half_h + BALL_R)
        )
        past = (ball_vel[1] < 0.0) & (ball_pos[1] < cfg.paddle_y - 0.5)

        if not cfg.net:  # Phase 1: episode ends at contact
            contact = near
            reward_contact = jp.where(contact, cfg.contact_bonus, 0.0)
            reward = shaping + reward_contact
            done = jp.float32(contact | past)
            state.metrics.update(
                reward_shaping=shaping,
                reward_contact=reward_contact,
                interception=jp.float32(contact),
            )
        else:  # Phase 2: play on after contact; score when the ball lands
            # x-line shaping: x is force-free in flight, so where the ball
            # crosses the paddle plane is exact across bounces — landing-point
            # shaping parks the paddle where the ball touches down instead of
            # where it can be struck (probe: 0/40 contacts for a perfect chaser)
            x_at_plane = ball_pos[0] + ball_vel[0] * (cfg.paddle_y - ball_pos[1]) / jp.minimum(
                ball_vel[1], -0.1
            )
            # shape toward the intercept point (x-line, baseline) — without
            # the y term the paddle drifts off the baseline with zero gradient
            shaping = jp.where(
                incoming,
                -cfg.shaping_scale
                * jp.hypot(paddle_pos[0] - x_at_plane, paddle_pos[1] - cfg.paddle_y),
                0.0,
            )
            contact = (prev_vy < 0.0) & (ball_vel[1] > 0.0) & near
            # pace bonus: reward outgoing speed so swinging through the ball
            # beats the safe-touch local optimum (flat-face rebounds rarely
            # clear the net; the swing is what must be discovered)
            reward_contact = jp.where(
                contact, cfg.contact_bonus + 0.3 * jp.clip(ball_vel[1], 0.0, 10.0), 0.0
            )
            # a ball that dies rolling short would otherwise rattle out the clock
            dead = incoming & (ball_pos[2] <= BALL_R * 1.2) & (jp.abs(ball_vel[1]) < 1.5)
            landed = (ball_pos[2] <= BALL_R * 1.2) & (ball_vel[2] <= 0.0) & (ball_vel[1] > 0.0)
            crossed = ball_pos[1] > 0.3  # landing beyond the net line
            returned = landed & crossed
            reward_return = jp.where(returned, cfg.return_bonus, 0.0)
            prox = jp.clip(
                1.0
                - jp.linalg.norm(ball_pos[:2] - jp.array([0.0, cfg.target_y])) / cfg.target_radius,
                0.0,
                1.0,
            )
            reward_target = jp.where(returned, cfg.target_bonus * prox, 0.0)
            reward = shaping + reward_contact + reward_return + reward_target
            done = jp.float32(landed | past | dead)
            state.metrics.update(
                reward_shaping=shaping,
                reward_contact=reward_contact,
                interception=jp.float32(contact),
                reward_return=reward_return,
                reward_target=reward_target,
                returned=jp.float32(returned),
            )

        return state.replace(
            pipeline_state=ps, obs=self._obs(ps), reward=jp.float32(reward), done=done
        )

    def _obs(self, ps) -> jp.ndarray:
        parts = [ps.x.pos[1], ps.xd.vel[1], ps.x.pos[0][:2], ps.xd.vel[0][:2]]
        if self.cfg.orientation:
            parts.append(jp.array([ps.q[2], ps.qd[2]]))  # tilt angle, angular velocity
        return jp.concatenate(parts)
