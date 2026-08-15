# Teaching an agent to play tennis in JAX — Part 2: the return

> Target home: jamesponwith.github.io, alongside the Phase 2 gif.

Part 1 ended with a paddle that meets 95% of serves. Part 2 asks it to put
the ball back over the net into the far court, 70% of the time. Final:
**71.7% of 1000 random serves returned.**

The number is the least interesting part. It took six training runs, and
the agent cheated better than it played. My spec's risk register said reward
hacking would show up in curves before it showed up in gifs. It was right
three times.

## Hack #1: the phantom return

First eval, *random* policy: "returned 85%." A random paddle does not
return 85% of serves. The per-term metrics — logged from day one for
exactly this — showed the returns weren't returns: low serves smacked into
the far side of the net, rebounded, and landed beyond it, and my detector
counted them. Fix: serves must analytically clear the net at spawn, plus a
regression test. Honest baseline after fix: 0.0%.

## The run that couldn't

The first real run flatlined at 0%. Instead of shotgunning hyperparameters,
I wrote a 40-line probe — a scripted policy that chases the ball perfectly —
and asked: can *anyone* return a serve here? No. 0/40 contacts. Three
physical causes, found in order:

1. **The paddle face was dead.** I'd tuned restitution for the court and
   ball but never the paddle, which kept MuJoCo's critically-damped
   default: 8 m/s in, 1.5 m/s out. No return could reach the net. Ever.
2. **Serves hopped the paddle.** Net clearance made serves lobs; 21 of 30
   crossed above the paddle's 1-meter top. The paddle grew to player
   height.
3. **The shaping pointed at the wrong spot.** "Stand where the ball lands"
   parks you where it touches down — not where it can be struck. Nothing
   pushes the ball sideways in flight, so where it crosses the paddle's
   plane is exact, even across bounces. Shape toward that.

After the fixes, the scripted probe returned 20/40. Provably learnable.
PPO still refused — until bisection against Phase 1's known-good config
found the poison: **the paddle-tilt action channel.** Random exploration on
the tilt hinge lays the face flat often enough to destroy the contact
gradient everything else depends on. Dropped it. Flat-face swings clear the
net fine. (The wrist came back later as a curriculum — it buys *aim*, not
consistency. A story for Part 3.)

## Hack #3: wall-ball

The next run played honest tennis for four million steps — 79% returns —
then found something better than tennis. The contact metric read **3170%**:
thirty-one contact events per episode. The policy had learned to drive to
the net, pin the ball against it, and rattle it off the paddle face,
farming the contact bonus into a reward of +282 while real returns crashed
to 17%. My spec had predicted "micro-touches that maximize contact bonus
without returning" almost verbatim.

The fix was mechanical, not reward surgery: a joint limit keeps the
receiver in the backcourt — where the honest 79% policy had been playing
anyway. Net-pinning became physically impossible. The regression test
charges the net for 1.2 seconds and asserts the paddle stays home.

## Run six: tennis

Contact ~97%, no farming, returns grinding through the plateau to **71.7%**
on the final 1000-serve eval. 10.9M env steps, about 90 minutes of laptop
CPU.

![returned rate](phase2_curve.png)

## What I'd tell past me

- Log per-term reward components before you need them. All three hacks were
  diagnosed from a single metric line each.
- When a run flatlines, write the scripted probe before touching
  hyperparameters. "Can a perfect policy even do this?" falsified my first
  three theories for the price of one script.
- Bisect from known-good instead of theorizing. The tilt channel was
  invisible to inspection and obvious to bisection.
- Prefer mechanical fixes — a joint limit, a taller paddle — over reward
  surgery. Physics can't be reward-hacked.

Next: [Part 3](writeup-phase3.md) — self-play, and the bug that beat every
probe.

*Built inside a [personal agentic flywheel](https://github.com/jamesponwith/agentic-flywheel)
— every fix above landed as a gated PR with tests. The repo is
[jamesponwith/brax-tennis-rl](https://github.com/jamesponwith/brax-tennis-rl).*
