# Relative XY 80% / yaw 45 degrees — approved 2026-10-16

User approval: implement B at 80%, yaw correction +/-45 degrees, direct feasible
action generation; preserve Hybrid, native ACT temporal integration and wrist-Z
admittance. Test, then execute one new target-1 layout in zero/continuous arms.
Deliver stacked video, wrist curves and IK evidence. NO training or Git push.

At each accepted 25 Hz ACT frame, p is the measured TCP XY before execution,
q is that frame's causally accepted, temporally integrated ACT target. Let d=q-p.
The Actor's two normalized XY coordinates map continuously from the square to
the unit disk: (x sqrt(1-y²/2), y sqrt(1-x²/2)). Interpret these coordinates in
the ACT-forward / perpendicular frame. Generate a correction of length at most
0.8 |d|. Any original workspace boundary further reduces the available radial
length BEFORE generation; never generate an invalid target then project it.
Reserve a few float64 coordinate ULPs inside that radius before generation so
sub-ULP requests become exact zero rather than rounding outside the 80% disk.
Yaw correction is normalized_action[2] * pi/4, with angular wrapping.

Zero raw action and zero filter state preserve q's original dtype and bytes.
When |d|=0, XY correction is exactly zero. Relative corrections are recomputed
from actual position every frame, not added to the previous correction. The
geometry implies final XY direction <=asin(.8)=53.13 degrees, satisfying the
requested forward half-plane without an extra 10-degree cone/progress limiter.

Retain 0.2-second smoothing of NORMALIZED action before feasible generation;
the old fixed 5mm/3-degree pose limits no longer define the execution path.
The generated target then passes through the UNCHANGED native 100 Hz reference
interpolator and its original total speed / wrist / workspace safety protections.
The relative bound is on the generated 25 Hz target, not on a post-slew-limit
100 Hz substep or the robot's achieved motion. Report these separately.

Ruling: do not subtract a historic pose offset at query submission or frame
start. B's correction is relative to the present actual state and accepted ACT
target; retaining the old pose-anchor subtraction would change this definition.
Cost if wrong: a different trajectory definition, caught by actual-anchor tests.

No accepted ACT chunk means no new correction, preserving native hold behavior.
No target truth, contact truth or extra image queries enter the Actor. Append
the accepted ACT displacement/yaw context as control observations, never a
future prediction. Old replay/checkpoints bind a different action contract and
must be rejected. Reward/update cadence/rollback protections otherwise remain.

Continuous preflight is a deterministic input signal, not a learned policy:
25 Hz requests with XY normalized amplitude .30 and yaw amplitude .10, smooth
one-second onset; XY period 4s, yaw period 5s. It runs until native termination.
Only one attempt per arm, shared native frame/window budgets, no retries.
The two independent closed-loop runs retain native asynchronous ACT timing;
the comparison is not a causal or statistical improvement claim.
