"""Rally env unit tests — mirroring, crossings, opponent actuation."""

import jax
import jax.numpy as jp
import pytest

from envs.rally import RALLY_CONFIG, RallyTennis

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def env() -> RallyTennis:
    e = RallyTennis()
    e.set_opponent(lambda obs: jp.zeros(2))
    return e


def _state(env, ball_pos, ball_vel=(0, 0, 0)):
    q = jp.concatenate(
        [jp.zeros(4), jp.array(ball_pos, dtype=jp.float32), jp.array([1.0, 0, 0, 0])]
    )
    qd = jp.concatenate([jp.zeros(4), jp.array(ball_vel, dtype=jp.float32), jp.zeros(3)])
    ps = env.pipeline_init(q, qd)
    state = env.reset(jax.random.PRNGKey(0))
    return state.replace(pipeline_state=ps, obs=env._obs(ps))


def test_sizes_match_phase2(env):
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (10,), "learner interface must stay warm-startable"
    assert env.sys.act_size() == 4  # learner 2 + opponent 2
    assert env.action_size == 2, "trainer must size the policy for the learner only"


def test_opponent_mirror_actuation(env):
    """Opponent's +y (toward the net, in its frame) is world -y."""
    env.set_opponent(lambda obs: jp.array([1.0, 1.0]))
    state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    for _ in range(25):
        state = step(state, jp.zeros(2))
    opp = state.pipeline_state.x.pos[0]
    assert opp[0] > 0.3, "opponent +x should be world +x"
    assert opp[1] < -RALLY_CONFIG.paddle_y - 0.3, "opponent +y should move it toward the net"
    env.set_opponent(lambda obs: jp.zeros(2))


def test_crossing_counts_both_directions(env):
    step = jax.jit(env.step)
    toward_far = _state(env, (0.0, -0.05, 1.5), (0.0, 6.0, 1.0))
    nxt = step(toward_far, jp.zeros(2))
    assert nxt.metrics["rally_length"] == 1.0
    assert nxt.metrics["returned"] == 1.0, "learner-direction crossing"
    toward_near = _state(env, (0.0, 0.05, 1.5), (0.0, -6.0, 1.0))
    nxt = step(toward_near, jp.zeros(2))
    assert nxt.metrics["rally_length"] == 1.0
    assert nxt.metrics["returned"] == 0.0, "opponent-direction crossing is not a learner return"


def test_rally_ends_when_ball_lands(env):
    state = _state(env, (0.0, 3.0, 0.05), (0.0, 2.0, -1.0))
    nxt = jax.jit(env.step)(state, jp.zeros(2))
    assert nxt.done == 1.0
