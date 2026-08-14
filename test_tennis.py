"""Env unit tests — spawn distribution, reward terms, terminations (SPEC:
env bugs masquerade as RL mysteries; test the env like normal code)."""

import jax
import jax.numpy as jp
import pytest

from envs.tennis import Tennis, TennisConfig, predict_landing

CFG = TennisConfig()


@pytest.fixture(scope="module")
def env() -> Tennis:
    return Tennis(CFG)


def _state_with(env: Tennis, ball_pos, ball_vel=(0, 0, 0), paddle_xy=(0, 0)):
    """Pipeline state with the ball/paddle placed by hand."""
    q = jp.concatenate(
        [
            jp.array(paddle_xy, dtype=jp.float32),
            jp.array(ball_pos, dtype=jp.float32),
            jp.array([1.0, 0, 0, 0]),
        ]
    )
    qd = jp.concatenate([jp.zeros(2), jp.array(ball_vel, dtype=jp.float32), jp.zeros(3)])
    ps = env.pipeline_init(q, qd)
    state = env.reset(jax.random.PRNGKey(0))
    return state.replace(pipeline_state=ps, obs=env._obs(ps))


def test_obs_and_action_sizes(env):
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (10,)
    assert env.action_size == 2


def test_serve_distribution(env):
    """100 serves: spawn in the far half, inside the cone, flying at the paddle."""
    keys = jax.random.split(jax.random.PRNGKey(1), 100)
    obs = jax.vmap(env.reset)(keys).obs
    ball_pos, ball_vel = obs[:, 0:3], obs[:, 3:6]
    assert jp.all(jp.abs(ball_pos[:, 0]) <= CFG.court_half_w + 1e-5)
    assert jp.all(ball_pos[:, 1] == CFG.serve_y)
    assert jp.all((ball_pos[:, 2] >= CFG.serve_h_lo) & (ball_pos[:, 2] <= CFG.serve_h_hi))
    assert jp.all(ball_vel[:, 1] < 0), "every serve must fly toward the paddle"
    speeds = jp.linalg.norm(ball_vel[:, :2], axis=1)
    assert jp.all((speeds >= CFG.serve_speed_lo - 1e-4) & (speeds <= CFG.serve_speed_hi + 1e-4))
    # spawns actually vary
    assert jp.std(ball_pos[:, 0]) > 0.5


def test_contact_gives_bonus_and_ends_episode(env):
    # ball just in front of the face, flying into it — one step spans the impact
    state = _state_with(env, ball_pos=(0.0, CFG.paddle_y + 0.3, 0.5), ball_vel=(0, -8, 0))
    nxt = jax.jit(env.step)(state, jp.zeros(2))
    assert nxt.metrics["reward_contact"] == CFG.contact_bonus
    assert nxt.metrics["interception"] == 1.0
    assert nxt.done == 1.0
    assert nxt.reward > CFG.contact_bonus - 1.0  # bonus dominates shaping


def test_ball_past_paddle_ends_without_bonus(env):
    state = _state_with(env, ball_pos=(0.0, CFG.paddle_y - 2.0, 0.5), ball_vel=(0, -5, 0))
    nxt = jax.jit(env.step)(state, jp.zeros(2))
    assert nxt.done == 1.0
    assert nxt.metrics["reward_contact"] == 0.0
    assert nxt.metrics["interception"] == 0.0


def test_shaping_prefers_paddle_under_landing_point(env):
    """Reward is higher when the paddle sits where the ball will land."""
    ball = {"ball_pos": (2.0, 0.0, 2.0), "ball_vel": (0.0, -8.0, 0.0)}
    landing = predict_landing(jp.array(ball["ball_pos"]), jp.array(ball["ball_vel"]))
    near = jax.jit(env.step)(
        _state_with(env, **ball, paddle_xy=(landing[0], landing[1] - CFG.paddle_y)), jp.zeros(2)
    )
    far = jax.jit(env.step)(_state_with(env, **ball, paddle_xy=(-3.0, 0.0)), jp.zeros(2))
    assert near.metrics["reward_shaping"] > far.metrics["reward_shaping"]


def test_predict_landing_math():
    # straight drop from 1 m: lands where it is
    flat = predict_landing(jp.array([1.0, 2.0, 1.0]), jp.array([0.0, 0.0, 0.0]))
    assert jp.allclose(flat, jp.array([1.0, 2.0]))
    # with horizontal velocity: x advances by vx * fall time
    t = jp.sqrt(2 * (1.0 - 0.033) / 9.81)
    moving = predict_landing(jp.array([0.0, 0.0, 1.0]), jp.array([3.0, 0.0, 0.0]))
    assert jp.allclose(moving[0], 3.0 * t, atol=1e-3)


def test_action_moves_paddle(env):
    state = env.reset(jax.random.PRNGKey(2))
    step = jax.jit(env.step)
    for _ in range(25):  # half a second at full +x
        state = step(state, jp.array([1.0, 0.0]))
    assert state.obs[6] > 0.5, "paddle should have moved right"


# ---- Phase 2 ----

P2 = TennisConfig(net=True, orientation=True, serve_h_lo=1.8, serve_h_hi=3.0, paddle_half_h=1.0)


@pytest.fixture(scope="module")
def env2() -> Tennis:
    return Tennis(P2)


def _state2(env2: Tennis, ball_pos, ball_vel=(0, 0, 0)):
    q = jp.concatenate(
        [jp.zeros(3), jp.array(ball_pos, dtype=jp.float32), jp.array([1.0, 0, 0, 0])]
    )
    qd = jp.concatenate([jp.zeros(3), jp.array(ball_vel, dtype=jp.float32), jp.zeros(3)])
    ps = env2.pipeline_init(q, qd)
    state = env2.reset(jax.random.PRNGKey(0))
    return state.replace(pipeline_state=ps, obs=env2._obs(ps))


def test_phase2_sizes(env2):
    state = env2.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (12,)
    assert env2.action_size == 3


def test_net_blocks_low_ball(env2):
    """A ball flying at the net below net height gets stopped; without a net it crosses."""
    start = (0.0, -1.5, 0.4)
    vel = (0.0, 8.0, 0.0)
    step = jax.jit(env2.step)
    state = _state2(env2, start, vel)
    for _ in range(20):  # 0.4 s — plenty to travel 1.5 m at 8 m/s
        state = step(state, jp.zeros(3))
    assert state.obs[1] < 0.5, "net should stop a below-net-height ball"


def test_return_landing_scores(env2):
    """Post-hit ball (vy > 0) landing near the target center earns both bonuses."""
    state = _state2(env2, (0.0, P2.target_y - 0.1, 0.05), (0.0, 2.0, -1.0))
    nxt = jax.jit(env2.step)(state, jp.zeros(3))
    assert nxt.done == 1.0
    assert nxt.metrics["returned"] == 1.0
    assert nxt.metrics["reward_return"] == P2.return_bonus
    assert nxt.metrics["reward_target"] > 0.5 * P2.target_bonus


def test_incoming_ball_landing_does_not_score(env2):
    """A ball still flying -y (never hit) landing on the far half scores nothing."""
    state = _state2(env2, (0.0, P2.target_y, 0.05), (0.0, -2.0, -1.0))
    nxt = jax.jit(env2.step)(state, jp.zeros(3))
    assert nxt.metrics["returned"] == 0.0
    assert nxt.metrics["reward_return"] == 0.0
    assert nxt.done == 0.0, "incoming bounces are not terminal"


def test_contact_event_needs_vy_flip(env2):
    """Sitting near the paddle without a -to-+ vy flip is not a contact event."""
    state = _state2(env2, (0.0, P2.paddle_y, 0.5), (0.0, 1.0, 0.0))  # already outgoing
    nxt = jax.jit(env2.step)(state, jp.zeros(3))
    assert nxt.metrics["interception"] == 0.0


def test_phase2_serves_clear_net(env2):
    """Every serve's ballistic arc clears the net — else net rebounds fake returns."""
    keys = jax.random.split(jax.random.PRNGKey(3), 100)
    obs = jax.vmap(env2.reset)(keys).obs
    h, vy, vz = obs[:, 2], obs[:, 4], obs[:, 5]
    t_net = P2.serve_y / -vy
    z_at_net = h + vz * t_net - 0.5 * 9.81 * t_net**2
    assert jp.all(z_at_net > P2.net_height), "a serve would hit the net"
