"""Phase 3: rally self-play — two paddles, one learner, one frozen opponent.

The learner keeps Phase 2's exact interface (obs 10, act 2, world frame), so
a Phase 2 checkpoint warm-starts with no surgery; the far paddle is driven
inside `step` by a frozen policy fed MIRRORED observations (y and vy negated
— the opponent believes it is the near paddle). Court symmetry makes one
policy playable from both ends.

Rally length rides the evaluator the same way `interception` does: the
`rally_length` metric is 1.0 only on a step where the ball crosses the net,
so the episode sum IS the number of crossings. Success (SPEC Phase 3):
average rally length grows over training.

Reward: Phase 2's incoming shaping + contact/pace bonus, plus a crossing
bonus each time the learner's shot crosses the net. The episode ends when
the ball lands (either side — rally over), passes either baseline, or dies.

Scene indices (rally=True): link 0 = opponent, 1 = learner paddle, 2 = ball;
ctrl = [learner x, learner y, opponent x, opponent y].
"""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jp
from brax.envs.base import State

from envs.tennis import BALL_R, PHASE2_CONFIG, Tennis, TennisConfig

RALLY_CONFIG = dataclasses.replace(PHASE2_CONFIG, rally=True)

_MIRROR3 = jp.array([1.0, -1.0, 1.0])  # reflect across the net plane
_MIRROR2 = jp.array([1.0, -1.0])


class RallyTennis(Tennis):
    """Two-paddle court. Call set_opponent() before stepping."""

    def __init__(self, config: TennisConfig | None = None):
        super().__init__(config or RALLY_CONFIG)
        self._opp_fn: Callable | None = None

    @property
    def action_size(self) -> int:
        return 2  # learner paddle only — the opponent's ctrl is driven internally

    def set_opponent(self, fn: Callable) -> None:
        """fn: (obs10 in the opponent's mirrored frame) -> act2 in [-1, 1]."""
        self._opp_fn = fn

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        state.metrics.update(rally_length=0.0, reward_cross=0.0)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        cfg = self.cfg
        prev_ball_y = state.obs[1]
        prev_vy = state.obs[4]

        # opponent sees the world reflected: it thinks it is the near paddle
        ps0 = state.pipeline_state
        opp_obs = jp.concatenate(
            [
                ps0.x.pos[2] * _MIRROR3,
                ps0.xd.vel[2] * _MIRROR3,
                ps0.x.pos[0][:2] * _MIRROR2,
                ps0.xd.vel[0][:2] * _MIRROR2,
            ]
        )
        opp_act = self._opp_fn(opp_obs)
        opp_ctrl = jp.clip(opp_act, -1.0, 1.0) * _MIRROR2 * cfg.max_paddle_speed
        ctrl = jp.concatenate([jp.clip(action, -1.0, 1.0) * cfg.max_paddle_speed, opp_ctrl])
        ps = self.pipeline_step(ps0, ctrl)

        ball_pos, ball_vel = ps.x.pos[2], ps.xd.vel[2]
        paddle_pos = ps.x.pos[1]

        shaping, _, contact, reward_contact = self._strike(ball_pos, ball_vel, paddle_pos, prev_vy)

        crossing = jp.sign(ball_pos[1]) != jp.sign(prev_ball_y)
        learner_cross = crossing & (ball_vel[1] > 0.0)
        reward_cross = jp.where(learner_cross, cfg.return_bonus, 0.0)

        landed = (ball_pos[2] <= BALL_R * 1.2) & (ball_vel[2] <= 0.0)
        past_near = (ball_vel[1] < 0.0) & (ball_pos[1] < cfg.paddle_y - 0.5)
        past_far = (ball_vel[1] > 0.0) & (ball_pos[1] > -cfg.paddle_y + 0.5)
        dead = (ball_pos[2] <= BALL_R * 1.2) & (jp.abs(ball_vel[1]) < 1.5)
        done = jp.float32(landed | past_near | past_far | dead)

        reward = shaping + reward_contact + reward_cross
        state.metrics.update(
            reward_shaping=shaping,
            reward_contact=reward_contact,
            interception=jp.float32(contact),
            reward_return=jp.float32(0.0),
            reward_target=jp.float32(0.0),
            returned=jp.float32(learner_cross),
            rally_length=jp.float32(crossing),
            reward_cross=reward_cross,
        )
        return state.replace(
            pipeline_state=ps, obs=self._obs(ps), reward=jp.float32(reward), done=done
        )

    def _obs(self, ps) -> jp.ndarray:
        # learner obs identical to Phase 2: ball + own paddle, world frame.
        # ponytail: the opponent is invisible — the ball says enough for v1.
        return jp.concatenate([ps.x.pos[2], ps.xd.vel[2], ps.x.pos[1][:2], ps.xd.vel[1][:2]])
