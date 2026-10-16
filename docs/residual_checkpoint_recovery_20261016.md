# Authorized recovery of the 100,639-step run

Scope: checkpoint persistence and explicit continuation only. Hybrid, frozen ACT,
temporal integration, observation/action/reward contracts, Z admittance, learning
cadence, policy guard thresholds and resource limits are unchanged.

The original full checkpoint tried to pickle a growing NumPy replay snapshot in
one large `data.pkl` record. The first error was `checkpoint deadline exhausted`;
PyTorch ZIP finalization masked it with `unexpected pos`. Tensor record streaming
removed the large pickle but the full replay rewrite still exceeded the same
55-second window on this machine.

Nonempty full checkpoints now use `durable_records_v1`: model/Adam/RNG state is
inline, with exact replay membership, sampler state and hash-checked references
to already-fsynced episode `experience.npz` blocks. Existing NumPy and tensor
snapshots remain readable. Reference checkpoints require the run directory and
its referenced files for training recovery; Actor inference needs no replay.
Save/reuse verification checks the numerical dependencies as well as the `.pt`.
Writer failures retain their first cause, and partial temporary files belonging
to the current write are removed without touching completed checkpoints.

The original compact `interrupted.pt` was validated at 100639 effective steps,
1225 Critic updates and 225 Actor updates. All 142 committed episodes/experience
hashes match its ledger prefix. Reconstructing sampler draws and update permits
from the real historical masks matches the independent 50K full checkpoint's
RNG, cursor, membership and update count exactly. The final reconstructed replay
contains 111 episodes / 99766 total context steps, cursor 4, schedule remainder 6.
The fixed policy probe histories are rebuilt from their original episodes, even
if those episodes have left the live replay pool. No physical action or learning
update is replayed during this reconstruction.

`resume_authorization.json` binds the original seal/source hashes, the exact
reviewed persistence-only changes, checkpoint and ledger. Original seals are not
rewritten. A pre-resume archive preserves status/launch/error/shutdown evidence
and hard-links every completed pre-resume checkpoint. Full recovery validation
precedes creation of the one-use `resume_started.json` marker; the worker checks
again. The next opportunity is `train_0000142`, seed 5000142, target 5.

Continuation appends existing metrics/ledger, restores model/Adam/local and global
RNGs, replay sampler and update permissions, restores consumed-layout registry,
and keeps the original monotonic start and 1M/72h/20GiB/8GiB limits. Shutdown
acknowledgements are bound to the new attempt token. There is no automatic retry,
preflight, six-layout validation, new initial model, or restart at zero.

Explicit entry (only after authorization/verification):

```sh
.venv/bin/python -B -u scripts/train_residual_long.py --mode resume \
  --out /absolute/existing/run --acknowledge-training
```
