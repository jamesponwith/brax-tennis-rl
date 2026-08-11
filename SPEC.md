# Brax Tennis RL Agent — Spec

**One-liner:** Train a PPO agent in Brax (Google's JAX physics engine) to intercept and return a
tennis ball, on free/cheap accelerators, with rendered rollouts and a writeup.

**Why this project:** DeepMind is a JAX shop and an RL house; this is a project *in their stack*.
Combined with the NCAA D1 tennis background, it's the application's memorable story: the collegiate
tennis champion who taught an agent to play. The rendered gif also refreshes the website's dated
ML content (PoseNet/CNN era → JAX/RL era).

## Scope philosophy

RL projects die from environment-engineering scope creep. Phases are strictly ordered, each with a
kill-switch deliverable: **if the project stops after any phase, that phase's artifact is already
publishable.** Phase 1 alone is a success.

## Phases

### Phase 0 — Validate the training loop (1 evening)
- Colab (free tier first; pay for A100 hours only when Phase 2 needs them).
- Run Brax's stock PPO on `ant`: confirm training curves, checkpointing, HTML/gif rendering of
  rollouts end-to-end.
- Deliverable: notebook that trains ant and renders a rollout. No custom code beyond glue.
- Purpose: every later debugging session starts from a known-good loop.

### Phase 1 — Ball interception (the core)
Custom Brax env, deliberately minimal bodies:
- **World**: a plane (court), an agent — a paddle modeled as an actuated box (position/velocity
  control in x, y), no articulated arm, no humanoid. The paddle IS the player.
- **Ball**: sphere spawned at random position/velocity within a serve-like cone toward the agent's
  half. Ballistic + bounce (Brax handles restitution). No spin (see Non-goals).
- **Observation**: ball position (3), ball velocity (3), paddle position (2), paddle velocity (2)
  → 10-dim vector. No pixels — state-based first, always.
- **Action**: paddle target velocity (2-dim continuous).
- **Reward**: dense shaping — negative distance from paddle to ball's landing point each step;
  +10 on contact; episode ends on contact, ball-past-paddle, or timeout.
- **Success metric**: interception rate ≥ 95% over 1000 random serves.
- Deliverable: training curve + gif of the paddle sliding to meet serves. Publishable alone.

### Phase 2 — The return
- Add a net (static box with collision) and a target region on the far half.
- Reward: Phase 1 shaping + return bonus: ball crosses net after contact (+10), lands in target
  region (+10 scaled by proximity to center).
- Contact quality now matters: paddle angle/velocity at impact determines return trajectory —
  add paddle orientation (1-dim, rotation about the horizontal axis perpendicular to travel) to
  the action space, giving flat-through-lob control. Re-tune from a Phase 1 checkpoint.
- Watch for reward hacking (e.g., micro-touches that maximize contact bonus without returning);
  log per-term reward components from day one so hacks are visible in the curves.
- **Success metric**: ≥ 70% of serves returned into the valid far court.
- Deliverable: gif of serve → slide → return over the net. This is the money shot.

### Phase 3 (stretch) — Rally self-play
- Mirror the env: two paddles, ball alternates sides; self-play against past checkpoints
  (league of 3–5 snapshots, sampled opponents).
- Success = average rally length grows over training.
- Explicitly optional. Do not start unless Phase 2 is written up and on the website.

## Tech choices

- **Brax v2, MJX/positional backend** — pick one early via a bounce-fidelity smoke test
  (ball restitution behaves) and stick with it; backend-hopping is a classic time sink.
- **PPO from `brax.training`** — do NOT write PPO in Phase 0–2; env design is the project.
  A from-scratch PPO (in the same repo, `ppo_scratch/`) is a Phase-3-adjacent stretch goal purely
  for the learning writeup.
- **Env structure**: single `envs/tennis.py` with a config dataclass (phase toggles: net on/off,
  target region, orientation control) rather than three env classes.
- **Experiment tracking**: wandb free tier (curves in the writeup come from here); seeds fixed and
  logged; every gif traceable to a checkpoint + config.
- **Repo**: uv for deps, ruff, pytest for env unit tests (spawn distribution, reward terms,
  termination conditions — env bugs masquerade as RL mysteries; test the env like normal code).

## Compute budget

- Phase 0–1: Colab free / local CPU (Brax trains simple envs in minutes on a single accelerator).
- Phase 2: a few hours of Colab Pro A100 (~$10–20 total).
- If a run needs > 4 GPU-hours, the env is wrong — simplify, don't scale.

## Deliverables & writeup

1. Repo: env + training scripts + tests + configs, runnable end-to-end from README.
2. Rendered gifs per phase (Brax HTML renderer → screen-record → gif).
3. Writeup: "Teaching an agent to play tennis in JAX" — env design decisions, reward-shaping
   failures (document the hacks — reviewers love honest negative results), curves, what Brax made
   easy/hard. Cross-link the D1 tennis background explicitly.
4. Website: new project card (tags: tech + sport) with the Phase 2 gif; retire or demote the
   PoseNet card when this lands.

## Non-goals

- Spin/Magnus effect, realistic racket strings, humanoid players, pixel observations, sim-to-real.
- Beating any benchmark. The artifact is the env design + honest writeup, not SOTA.
- Tournament-grade self-play infrastructure.

## Risks

| Risk | Mitigation |
|---|---|
| Env engineering rabbit hole | Phase gates + kill-switch deliverables; 4-GPU-hour rule |
| Reward hacking stalls Phase 2 | Per-term reward logging from day one; shaping changes are ADRs |
| Brax API churn / backend quirks | Pin versions in Phase 0; bounce smoke test before building |
| Motivation cliff after Phase 1 | Phase 1 is already publishable; ship its writeup immediately |
