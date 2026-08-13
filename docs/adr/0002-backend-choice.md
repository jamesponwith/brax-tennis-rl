# 0002: MJX backend, pinned versions

## Context

SPEC requires picking a physics backend early via a bounce-fidelity smoke
test (`tools/bounce_smoke.py`) — backend-hopping is a classic time sink.
Candidates: brax `mjx` vs `positional`.

## Decision

**MJX**, on brax 0.14.2 / jax 0.11.0 / mujoco 3.11.0 (pinned by uv.lock).

- positional ignores contact restitution outright: the dropped ball never
  leaves the floor. Disqualified.
- brax itself warns its non-MJX pipelines are no longer actively maintained
  and points at MJX.
- MJX restitution is tunable via `solref` dampratio. Operating point
  `solref="0.02 0.15"` at timestep 0.004 → rebound ≈ 33% of drop height
  (CoR ≈ 0.57, vs ≈ 0.73 for a real tennis ball).

## Consequences

Good enough for interception (Phase 1); the CoR ceiling is a known
compromise for Phase 2 return arcs. **dampratio < 0.15 is forbidden**: the
solver injects energy (1 m drop rebounds 1.9 m at 0.1). If Phase 2 needs a
livelier ball, tune timestep down before touching dampratio, and re-run the
smoke test.
