# Pre-DINO trial-4 Hybrid restoration

User-authorized scope: restore the independent ResNet Hybrid baseline, run ONE
target1 without the RL adapter, reconnect RL only after a normal restored run,
and publish the current residual-development branch. Do not start training.

## Provenance and preservation

- Reference: `feat/hybrid-astar-act-trial4`, commit
  `eb887843c9849d4a72be116bb87c44519dc7ef74`, tree
  `48a43be83dca4b63e547683f8f18d74bcce8fdb6` (GitHub API read-back).
- Simulator base: `9102680d8e018b225ab34d36422572cd4b547166`.
  All67 `sim/**/*.py` bytes match the reference and original frozen manifest.
- The original formal batch entry and14 runtime dependencies are retained
  byte-for-byte in `hybrid_trial4_snapshot/`, with their source manifest.
  Execution on this Mac still verifies/reuses the original historical absolute
  dependency paths. This restoration does not claim cross-machine portability.
- Original models, normalization, datasets, failed results, formal ledgers,
  plan/config/metadata/seal and stopped RL attempts are not rewritten.
- DINO-only source/config/test/guide files are removed from this execution
  branch, but preserved in the DINO branch and prior Git history. Existing RL
  learning/control modules are retained; this is not a hard reset or force push.

## Explicit original-Hybrid entry

The workbench/default `sim.act.evaluate` is still the underlying pure-ACT entry.
Use the explicit wrapper for the full A* prefix and fourth-group handoff:

```sh
# Read-only: no torch/MuJoCo/model loading, RL import, or physical execution.
.venv/bin/python -B scripts/verify_restored_hybrid.py

# One opportunity only; must use a new output directory and new seed.
.venv/bin/python -B scripts/verify_restored_hybrid.py --run-target1 \
  --seed 3152200 --out /absolute/exports/hybrid_restore_goal1_20261015
```

The wrapper calls the unchanged `prepare_contact` and
`build_formal_act_driver`, not `residual_rl.runtime` or its Collector. It retains
native temporal ensembling0.01 and all original controller/termination rules.
The first action's2-second50mm/s² handoff is unchanged; Z is wrist-admittance
plus bounded actual-Z following throughout, never ACT predicted Z.

Outputs include the real count/outcome, handoff and Z ownership audit, exact
integration-state replay, and a completely decoded/hashed MP4. Video generation
restores saved states, never reruns ACT. Images and video are retained for review.
An existing output directory is rejected; exceptions preserve partial state and
known actual quantity without converting a program error into a policy failure.

`restoration_verified` requires ACT actually executed beyond the handoff window,
native sweeping restored, no recorded controller failure/safety termination,
no policy-owned Z rows, and no handoff-limit violation. It does not require an
exact final quantity and is not a claim of continuous table contact or efficacy.

## Test boundaries

The published snapshot tests keep original relative/historical path assumptions.
Running those archived test files directly under their new snapshot location
produces7 path/missing-helper failures; their39 original numeric contracts can
instead be run at the original verified exports paths. They are not portable
execution tests. Production `tests/test_hybrid_restore.py` verifies the archived
bytes and the active original runtime, including no-RL inspection and the gate.

Six existing `tests/test_hybrid_controller.py` FakePlant/default0.1N tests fail:
`test_full_phase_sequence_is_visited`,
`test_contact_force_is_bounded_and_positive_while_sweeping`,
`test_force_is_released_before_the_guard_line`,
`test_force_sensor_noise_does_not_destabilise_the_loop`,
`test_transfer_and_retract_happen_clear_of_the_table`, and
`test_table_height_offset_is_absorbed_by_the_contact_search`.
Their tests, controller/config dependencies and default YAML are unchanged from
the pre-restoration HEAD. Do not retune the frozen1N Hybrid to fit those tests.
No whole-project green result is implied by focused tests or one rollout.

## Actual no-RL check: safety stop, RL reconnection held

The single authorized new layout used seed3152200 in
`exports/hybrid_restore_goal1_20261015`. No residual package was imported.
The real outcome is failure, final collected0, one opportunity / one physical
execution. A* contact preparation succeeded and ACT executed2869 control steps:
200 handoff-limited steps followed by2669 exactly native steps. There were no
handoff limit violations, rejected advanced rows, invalid samples, failed
control steps, or policy-owned Z rows.

The original wrist-force safeguard terminated the episode at simulation
time33.810s. Peak filtered wrist force29.875259N occurred in native sweeping;
handoff-limited peak was5.902357N. Mean ACT filtered wrist force1.077587N is
not proof of continuous table contact. At the peak, actual TCP Z was0.192217mm
and commanded Z1.953300mm; the posthandoff maximum Z following error was1.761082mm.

`restoration_verified=false`: source restoration is verified, but a normal
physical result is NOT established. Do not reconnect/execute the residual or
start training on the basis of this check. Do not retry this layout, replace its
failure or retune the preserved controller without a new user decision.
Physical result, numerical telemetry and saved-state replay are retained.
The verified video has836 frames,25fps,960x640 pixels and1,703,442 bytes;
see local `media.json` for its full SHA256 and retained replay path. No video
or frame images were deleted by this check.

Verification:115 focused tests passed after the two reviewed reporting fixes.
The bounded whole-project command completed167 tests with0 failures and1 warning
then was interrupted at153.75s (the150s bound plus interrupt handling). It did not
complete the full suite; the six separately observed FakePlant failures above
remain explicitly reported. No convergence, policy improvement or restored
contact stability is claimed.
The residual/restoration regression selection separately passed282 tests.
The archived `phases.py` retains its original extra blank line at EOF to preserve
the published SHA256; it is not reformatted merely to silence `git diff --check`.
