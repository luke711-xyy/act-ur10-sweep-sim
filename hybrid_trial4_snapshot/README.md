# Independent Hybrid source archive — not an execution entry

The independent reference branch is `feat/hybrid-astar-act-trial4`, commit
`eb887843c9849d4a72be116bb87c44519dc7ef74`. Its source/configuration hashes and
previous verification report are preserved here for provenance, not as a
claim about DINO performance.

On this DINO branch the original `scripts/evaluate_hybrid_paired_batch.py`
is archived here as `evaluate_hybrid_paired_batch.py`. The original manifest's
path refers to its location on the independent branch. Historical paths,
old checkpoint names, fixed seeds and test-probe mains are documentary only;
**do not execute these archived entry points**. They do not select DINO.

The portable active code is `sim/act/hybrid_trial4/`; the public CLI/workbench
entry is `python -m sim.act.evaluate`. DINO checkpoints select Hybrid there.
All 14 helper function/class ASTs and the four formal driver/configuration
adapters are checked against this archive. Only module-level path wiring is
relocated; no outside `exports` folder or historical batch is needed.

The control-profile JSON has the same values; GitHub preserves the original
archive blob including its no-final-newline formatting. The active profile is
a separately packaged JSON resource. See `docs/hybrid-dinov3-guide.md`.
