# Native 640x480 parallel demonstration collection

User authorization: replace current square captures with native dual-view 640x480;
accelerate regeneration with parallel workers while training is paused.

Scope: preserve the frozen 300 actual target/seed pairs, camera transforms,
expert candidate screening, controller, observation/action semantics and lossless
schema-v5 writer. Use a separate result root. Do not change daily inference or
the original dataset. Retain the 60 committed square episodes during replacement.

Implementation sequence (native execution in this worktree):

- [x] Derive collection image size from native render dimensions; test only image
  configuration changes, with controller/planner unchanged.
- [x] Add an opt-in disk-backed uint8 frame store to the capture loop. Test exact
  pixel equality, bounded allocation and capacity errors before implementation.
  The default capture path has no store and remains unchanged.
- [x] Add a bounded process coordinator using the existing manifest lock/writer.
  Each child owns one immutable slot and temporary image store. Parent records
  progress and failures; unexpected child errors stop all workers without retry.
  Committed episodes are recovered before scheduling. Test unique assignment,
  crash/error propagation, source identity, and completion accounting.
- [x] Reuse the first single-worker native episode as launch/size evidence.
  Start two disk-backed workers on the remaining formal slots; measure RAM,
  progress and throughput before choosing a higher worker count.
- [x] Verify real saved images are 640x480, numeric contracts remain intact and
  the original source manifest is unchanged. Report actual running progress,
  storage estimate and any remaining quota failures honestly.

Execution evidence: 43 scoped tests passed. Serial native episode seed4200256
saved 803 frames/1606 images, with all numeric arrays identical to its prior
square capture. After two concurrent disk-backed captures held near 1.5GiB,
the worker count increased to three and then four. First parallel commit
seed4200000 saved 994 frames; child peak RSS804644KiB. Resource sampling covers
encoding as well as capture; service total peak remained below its 3GiB cap,
with no OOM events. Timing varies by layout and phase, so no fixed speedup
factor is claimed. Full 300-slot collection remains a running background job.

Resources: single candidate-validation worker inside each episode worker; one
BLAS/OpenMP thread each. Explicit service memory cap, no swap, disk headroom
guard and no automatic retries. Do not infer completion from process liveness.
