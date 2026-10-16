# Bounded residual pose offsets — approved 2026-10-16

User authorization: implement and test only. No training, actual ACT preflight,
physical rollout, resume, checkpoint publication, Git commit or push in this task.
Supersedes the old velocity-action interface, not the Hybrid or learning schedule.

## Contract

The three normalized SAC outputs request the current desired world-XY offset
(radially bounded to 0.005 m) and yaw offset (bounded to +/- pi/60 rad). They are
not per-step increments. Z is copied bit-exact from the original controller.
ACT still queries at 5 Hz, decides at 25 Hz and executes at 100 Hz; temporal
ensembling, A* acquisition, original 2 s handoff, force feedback and safety stay.

The adapter removes its own applied XY/yaw offset from the reference captured
at query SUBMISSION and frame start. It never subtracts a later offset when a
future resolves. Thus corrected executed references cannot be re-used as fresh
uncorrected anchors and compound the same offset. This bounds the correction
relative to the causal, uncorrected ACT reference, not a hypothetical separate
closed-loop zero-residual episode, which could observe different images.

Implementation defaults for unspecified smoothing: 0.2 s first-order response,
XY offset rate <= 0.005 m/s and yaw offset rate <= pi/60 rad/s. No new limit is
applied to native ACT acceleration. During native XY motion the executed XY
velocity retains >= 80% of native forward progress, is within a 10-degree cone,
and obeys the original 0.08 m/s total bound. Native XY idle freezes the XY offset
instead of introducing motion; yaw may adjust while ACT is otherwise moving.
When all native XY/yaw motion stops both offsets freeze (no forced snap-to-zero).
Native yaw motion retains its direction and >=80% progress, with total 0.35 rad/s.

Project each proposed XY offset onto the intersection of the 5 mm disk, offset
rate disk, original workspace box, total-speed disk, progress and direction
halfplanes. Use deterministic 2-D boundary enumeration (including tangent disks)
to choose the nearest feasible point and VERIFY every constraint
after float32 output quantization; incompatible constraints raise a diagnostic
exception rather than silently relaxing limits, moving Z, retrying or holding.
Yaw is the corresponding scalar interval intersection, quantized to a
representable command value INSIDE that interval. Keep the exact original
float dtype/path when offset and action are zero. Pose rate limits, not an
unadvertised native acceleration limiter, define continuity in this revision.

When no native ACT chunk is available, the adapter recognizes the scheduler's
missing observation chunk and uses the unshifted frame start as the hold target.
It must not interpret native fallback's corrected last_action as a new nominal
target. Existing hold timing, rather than an additional query or retry, is kept.

## Integration and compatibility

Actor observations include both applied pose offset and its rate. Both are
control states, never object truth. Raw tanh actions and densities remain SAC's
action variables. Critic, masks, rewards, learning cadence and policy rollback
are otherwise unchanged. Policy-change checks use pose units, preserving the
old fractional thresholds (1/32 of XY bound, 1/28 of yaw bound).

New replay and learner schemas bind an explicit action contract and reject old
velocity-action snapshots even when network shapes happen to match. Probe stages
become small_offset / hold_offset / return_offset, not brake or reverse.

## Acceptance

Tests cover zero bit-exactness; long runs and repeated frame/query anchors;
asynchronous submission snapshots; radial XY/yaw limits; wraps; native stops and
turns; no slowdown-to-stop/reversal; full-speed saturation; workspace corners;
invalid inputs and infeasibility; controller state in Actor only; updated units
and checkpoint/replay rejection; original native scheduler/handoff parity.
CPU synthetic learner updates are unit tests, not an authorized training run.
