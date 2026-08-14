# Teaching an agent to play tennis in JAX — Part 2: the return

> **DRAFT** — structure and evidence in place; voice pass pending.
> Target home: jamesponwith.github.io, alongside the Phase 2 gif.

Part 1 ended with a paddle that meets 95% of serves. Part 2 asks it to do
something with the ball: put it back over the net into the far court, ≥70%
of the time. Final result: **71.7% of 1000 random serves returned** — but the
result is the least interesting part. It took six training runs, and the
agent cheated better than it played. This post is the debugging story,
because my spec's own risk register said reward hacking would show up in
curves before it showed up in gifs, and it was right three times.

## Hack #1: the phantom return (caught at eval #1)

First training eval, *random* policy: "returned 85%." A random paddle does
not return 85% of serves. The per-term reward metrics — logged from day one
precisely for this — showed the returns weren't returns: low serves were
smacking into the far side of the net, rebounding, and landing beyond it,
which my memoryless return detector happily counted. Fix: serves must
analytically clear the net at spawn, plus a 100-serve regression test.
Honest baseline after fix: 0.0%.

## The run that couldn't: physics forensics

The first real run flatlined at 0%. Instead of shotgunning hyperparameters,
I wrote a 40-line probe — a scripted policy that chases the ball perfectly —
and asked: can *anyone* return a serve in this environment? Answer: no.
0/40 contacts. Three physical causes, found in sequence:

1. **The paddle face was dead.** I'd tuned contact restitution for the court
   and ball but never the paddle geom, which got MuJoCo's critically-damped
   default: 8 m/s in, 1.5 m/s out. No rebound could reach the net, ever.
2. **Serves hopped the paddle.** Forcing serves to clear the net made them
   lobs; 21 of 30 crossed the paddle's plane above its 1-meter top. The
   paddle grew to player height.
3. **The shaping pointed at the wrong spot.** "Stand where the ball lands"
   parks you where it touches down — not where it can be struck. The fix
   uses the fact that nothing pushes the ball sideways in flight: where it
   crosses the paddle's plane is exact, even across bounces.

After the fixes the scripted probe returned 20/40. The task was provably
learnable. PPO still refused.

## The bisection: when in doubt, diff against known-good

Two more flat runs — entropy hypothesis: falsified by experiment; hinge
spring hypothesis: falsified by experiment. What worked was bisection: run
Phase 2's world with Phase 1's exact known-good configuration. That single
run learned contact to ~100% and returns to 73.6% in 2M steps. The delta
was one thing: **the paddle-tilt action channel**. Random exploration on the
tilt hinge lays the face flat often enough to destroy the contact gradient
everything else depends on. Dropped it; flat-face swings clear the net fine.
(Tilt returns someday as a curriculum from a flat-face checkpoint — that's a
Phase 3 problem.)

## Hack #3: wall-ball (the masterpiece)

The next run played honest tennis for four million steps — 79% returns —
and then found something better than tennis. The contact-rate metric read
**3170%**: thirty-one contact events per episode. The policy had learned to
drive to the net, pin the ball against it, and rattle it off the paddle
face, farming the +10 contact bonus into a reward of +282 while actual
returns crashed to 17%. My spec had predicted "micro-touches that maximize
contact bonus without returning" almost verbatim.

The fix was mechanical, not reward surgery: a joint limit keeps the
receiver in the backcourt — ±2m around the baseline, where the honest
79% policy had been playing anyway. Net-pinning became physically
impossible. The regression test charges the net for 1.2 seconds and asserts
the paddle stays home.

## Run six: tennis

Contact ~97%, no farming signature, returns grinding from 61% through the
plateau to **71.7%** on the final 1000-serve eval (recent evals 67–72% — a
pass at the line, reported as such). Total compute: 10.9M env steps on a
laptop CPU, about 90 minutes.

![returned rate](phase2_curve.png)

## What I'd tell past me

- Log per-term reward components before you need them. All three hacks were
  diagnosed from a single metric line each.
- When a run flatlines, write the scripted probe before touching
  hyperparameters. "Can a perfect policy even do this?" is one cheap script
  and it falsified my first three theories.
- Bisect from known-good instead of theorizing. The tilt channel was
  invisible to inspection and obvious to bisection.
- Prefer mechanical fixes (a joint limit, a spring, a taller paddle) over
  reward surgery. Physics can't be reward-hacked.

*Built inside a [personal agentic flywheel](https://github.com/jamesponwith/agentic-flywheel)
— every fix above landed as a gated PR with tests, and the local AI reviewer
caught a config bug that would have wasted an hour-long training run. The
repo is [jamesponwith/brax-tennis-rl](https://github.com/jamesponwith/brax-tennis-rl).*
