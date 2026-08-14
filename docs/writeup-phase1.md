# Teaching an agent to play tennis in JAX — Part 1: interception

> **DRAFT** — structure and evidence in place; voice pass pending.
> Target home: jamesponwith.github.io, alongside the Phase 1 gif.

I played NCAA D1 tennis. The first thing every coach teaches is not the
swing — it's *being where the ball will land*. So when I set out to train an
RL agent to play tennis in [Brax](https://github.com/google/brax) (Google's
JAX physics engine), Phase 1 was exactly that skill and nothing else:
**95% of random serves intercepted, or it doesn't count.**

Final result: **95.2% over 1000 serves**, stock PPO, 9.3 million env steps,
~20 minutes on a laptop CPU. This post is about the design decisions — and
the three things that went wrong before any learning happened.

## The paddle IS the player

RL projects die from environment-engineering scope creep, so the env is
deliberately minimal: a court plane, a served ball, and a paddle — an
actuated box sliding in x/y. No humanoid, no articulated arm, no pixels.
Observation is a 10-vector (ball position and velocity, paddle position and
velocity); action is a 2-vector of target paddle velocity.

The reward shaping carries the coaching insight directly: each step, the
penalty is the distance from the paddle to the ball's **predicted landing
point** — closed-form ballistics, re-predicted every step so bounces
self-correct — plus a +10 bonus on contact. Not distance-to-ball:
distance-to-where-the-ball-will-be. Split-step, read, move.

## Three things that broke before training started

**The backend that couldn't bounce.** Brax offers several physics backends;
the spec required a bounce-fidelity smoke test before committing. Good
thing: the `positional` backend ignores contact restitution entirely — a
ball dropped from 1m simply never leaves the floor. Disqualified by a
40-line script (and by brax's own deprecation warning pointing at MJX).

**The solver that broke conservation of energy.** Tuning MJX restitution:
contact `dampratio` 0.3 rebounds 19%, 0.15 rebounds 33%… and 0.1 rebounds
**190%** — the drop from 1m bounces to 1.9m, and at 0.05 to 37m. There is a
cliff where the solver starts injecting energy, and it sits just past the
setting you want. The operating point (CoR ≈ 0.57, a bit dead vs a real
ball's ≈ 0.73) and the "dampratio < 0.15 forbidden" rule are now an ADR.

**The trainer that outlived its dependency.** brax 0.14.2's PPO crashes on
jax ≥ 0.11 — `jax.device_put_replicated` is gone. The physics runs fine;
only training explodes. Phase 0's whole purpose was validating the loop
before writing custom code, and it caught exactly this. jax is pinned `<0.8`.

## Test the env like normal code

Env bugs masquerade as RL mysteries, so the env has unit tests like any
other module: serve-cone distribution, both termination paths, that reward
is higher with the paddle under the landing point, landing math against
analytic cases. They paid off immediately — brax's training wrappers inject
their own keys into the env's metrics dict, which surfaces as an opaque JAX
pytree error unless your `step` updates metrics in place. That's a
five-minute fix with a failing test and a lost evening without one.

Per-term reward components are logged from day one, because Phase 2's risk
register says reward hacking shows up in curves before it shows up in gifs.

## The curve

![interception rate](phase1_curve.png)

Random policy: 11.5%. Steep climb to ~4M steps, then a stable 94–95.6%
plateau for the final 3M. The metric is measured by the evaluator itself
over 1000 fresh serves per eval — the env emits `interception` only on the
contact step, so the episode-summed metric *is* the interception rate.

## Next

Phase 2 adds the net, a target region, and paddle orientation — contact
quality starts to matter, and the reward-hacking watch begins in earnest.
The money shot is a gif of serve → slide → return over the net.

*Built inside a [personal agentic flywheel](https://github.com/jamesponwith/agentic-flywheel):
spec-first, every change through a gated PR, public DORA metrics. The repo is
[jamesponwith/brax-tennis-rl](https://github.com/jamesponwith/brax-tennis-rl).*
