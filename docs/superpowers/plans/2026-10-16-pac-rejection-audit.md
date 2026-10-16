# PAC-ACT rejected-update audit and offline diagnosis

Authority: user approved audit completion/offline diagnosis, plus a separate
new-schedule no-noise120-layout paired evaluation. No PPO/preflight restart.
Base24f00bf. Existing sealed bundles/failed run/Hybrid/sim remain immutable.

1. Test and freeze standalone held-out baseline evaluator; launch with original
120 plan,100K weights,new synchronous5-step schedule,zero noise,no learning,
one opportunity per layout,real failure counts,live density and retained media.
Use old sealed bundle source so audit edits cannot change the running batch.
2. TDD rejected-update audit: preserve original traceback, stage/optimizer-step
metrics and exact masked distribution inputs; hash pre/proposed/restored model,
optimizer,noise,RNG,counters; verify rollback; atomically persist rejection
separately from accepted updates. Audit failure never suppresses rollback/error.
Wire preflight and GPU learner construction, without changing PPO parameters.
3. Offline only: verify KL formula against independent float64 Gaussian math;
analyze saved32 blocks,reward/GAE/noise/normalization and sensitivity to physical
mean shifts. Do not invent missing historical proposed means/rollback hashes.
No environment steps or adoption of offline-updated weights.
4. Focused regression suite plus full project suite; preserve/report old frozen
failures. Deliver findings,remaining uncertainty and parameter recommendations
for user confirmation; no automatic parameter edits or new training launch.

Interfaces: PPO audit -> append-only rejection store; frozen evaluation copy
-> independent source guard; recorded numeric blocks -> offline diagnostic only.
Review focus: audit failures must not prevent restore; nonfinite values must
remain JSON-safe; rejected metrics must not masquerade as accepted updates;
historical evidence stays immutable; zero-noise held-out evaluator cannot train.
