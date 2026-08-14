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
    state = _state_with(env, ball_pos=(0.0, CFG.paddle_y, 0.5))  # inside the paddle box
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
