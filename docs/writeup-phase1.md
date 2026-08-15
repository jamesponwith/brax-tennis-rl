# Teaching an agent to play tennis in JAX — Part 1: interception

> Target home: jamesponwith.github.io, alongside the Phase 1 gif.

I played NCAA D1 tennis. The first thing every coach teaches is not the
swing — it's being where the ball will land. So when I set out to train an
RL agent to play tennis in [Brax](https://github.com/google/brax), Phase 1
was that skill and nothing else: **95% of random serves intercepted, or it
doesn't count.**

Result: **95.2% over 1000 serves.** Stock PPO, 9.3M env steps, twenty
minutes on a laptop CPU. The interesting part is the three things that broke
before any learning happened.

## The paddle IS the player

RL projects die from environment-engineering scope creep, so the env is
deliberately minimal: a court, a served ball, and a paddle — an actuated box
sliding in x/y. No humanoid, no arm, no pixels. Ten numbers in, two out.

The reward carries the coaching insight directly: each step, the penalty is
the distance from the paddle to the ball's **predicted landing point** —
closed-form ballistics, re-predicted every step so bounces self-correct.
Not distance-to-ball. Distance-to-where-the-ball-will-be. Split-step, read,
move.

## Three things that broke first

**The backend that couldn't bounce.** My spec required a bounce-fidelity
smoke test before committing to a physics backend. Good thing: the
`positional` backend ignores restitution entirely — a ball dropped from 1m
never leaves the floor. Disqualified by a 40-line script.

**The solver that broke conservation of energy.** Tuning MJX restitution:
contact `dampratio` 0.3 rebounds 19%, 0.15 rebounds 33% — and 0.1 rebounds
**190%**. The 1m drop bounces to 1.9m; at 0.05, to 37m. There's a cliff
where the solver starts injecting energy, and it sits just past the setting
you want. The operating point and the "never below 0.15" rule are an ADR
now.

**The trainer that outlived its dependency.** brax 0.14's PPO crashes on
jax ≥ 0.11 — `jax.device_put_replicated` is gone. The physics runs; only
training explodes. Phase 0 existed to validate the loop before any custom
code, and it caught exactly this.

## Test the env like normal code

Env bugs masquerade as RL mysteries, so the env has unit tests: serve-cone
distribution, both termination paths, reward-is-higher-under-the-landing-
point, landing math against analytic cases. They paid off on day one —
brax's training wrappers inject keys into the env's metrics dict, which
surfaces as an opaque pytree error unless `step` updates metrics in place.
Five minutes with a failing test; a lost evening without one.

Per-term reward components are logged from day one, because Phase 2's risk
register says reward hacking shows up in curves before it shows up in gifs.
(It did. Three times. That's Part 2.)

## The curve

![interception rate](phase1_curve.png)

Random policy: 11.5%. Steep climb to ~4M steps, then a stable 94–95.6%
plateau. The metric rides the evaluator itself — the env emits
`interception` only on the contact step, so the episode-summed metric *is*
the interception rate over 1000 fresh serves.

Next: [Part 2](writeup-phase2.md) — the net goes up, and the agent starts
cheating.

*Built inside a [personal agentic flywheel](https://github.com/jamesponwith/agentic-flywheel).
The repo is [jamesponwith/brax-tennis-rl](https://github.com/jamesponwith/brax-tennis-rl).*
