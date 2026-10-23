# ACT Remote Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the complete existing ACT workbench privately, with remote training controls, inference and media.

**Architecture:** A Sites Worker serves a snapshot of the current UI and queues bounded HTTP requests in D1. A small local Python agent claims requests, forwards only allowlisted localhost APIs and uploads responses; R2 carries large responses. The agent uses a durable execution ledger so interrupted writes are never executed twice.

**Tech Stack:** Worker ESM, D1 SQLite, R2, Python standard library, Node built-in test/SQLite.

**Spec:** ../specs/2026-10-23-act-remote-site.md (approved by user 2026-10-23).

## Global Constraints

- ACT PID 339022, Diffusion PID 322477 and the 8773 workbench remain running and unchanged.
- Backend origin is fixed to http://127.0.0.1:8774; no arbitrary proxy or shell.
- Site remains owner-private; credentials never enter source, browser, argv, logs or ordinary config files.
- Writes execute at most once; offline/expired requests never execute after reconnect.
- No real training start/stop, inference or export for verification; isolated fake backend for writes.
- Independent source root: /home/luke/sites/rudra-act-workbench-20261023 (avoids embedded Git in managed training worktree).

## Review Focus

- Encoded traversal, alternate origins and redirects must not widen the allowlist.
- Disconnect after local write but before cloud acknowledgement must not repeat the write.
- Queued writes must expire when offline; a stale tab must display offline and disable controls.
- Large video/range responses must preserve content and statuses without loading unbounded files.
- Concurrent polling must stay bounded and preserve current training processes.

### Task 1: Cloud request queue and packaged workbench

**Files:** Create worker/index.js, worker/ui.js, drizzle/0000_bridge.sql, drizzle/meta/_journal.json, tests/worker.test.mjs, tests/d1.mjs, scripts/build.sh, .openai/hosting.json in independent source root.

**Interfaces:** Worker default.fetch(Request, env, ctx): Promise<Response>; env.DB D1 and env.BUCKET R2. Routes /remote/status, /bridge/claim, /bridge/complete/:id and existing workbench paths. Export allowed(method,path) for validation tests. Request fields id, method, path, body, headers, deadline; response fields status, headers, body or object_key.

- [ ] Write tests using actual Node SQLite for atomic claims, expired/offline writes, illegal paths, identity and origin requirements, inline/binary/range responses, and idempotent completion.
- [ ] Run `node --test tests/worker.test.mjs`; expect missing-feature failure.
- [ ] Implement schema-only migration and queue with 30-second write deadlines, one active agent, 15-second heartbeat freshness, read cache/dedup and bounded cleanup. Require browser identity plus same-origin writes; bridge uses confirmed private platform boundary.
- [ ] Copy currently served local UI/script without changing local sources. Add remote connection indicator, disable writes offline, reduce remote playback refresh and retain every existing control. Embed assets in final Worker build.
- [ ] Run same test command; expect all tests passing, then commit exact source.

### Task 2: Persistent local agent and once-only execution

**Files:** Create agent/bridge.py, tests/test_agent.py, scripts/inject-credential.py, docs/operations.md.

**Interfaces:** BridgeAgent.run_once() claims requests and posts responses. execute(request) returns status/headers/body. Local SQLite records id, state and response. Credential injected through owner-only Unix socket; Site origin in nonsecret service args.

- [ ] Write unittest cases against real isolated HTTP backend: start, stop and inference each called exactly once; repeated ID returns recorded result; interrupted executing state returns unknown; expired/invalid requests never reach backend; redirects refused; large files spool to disk and stream; credentials absent from files/output.
- [ ] Run `python3 -m unittest discover -s tests -p 'test_*.py'`; expect missing-feature failure.
- [ ] Implement standard-library agent with flock, SQLite FULL synchronization, fixed origin allowlist, bounded response sizes and thread pool, no mutating retries, response-only retries, and one-time socket credential injection.
- [ ] Run unittest and cloud suite; expect all passing, then commit.

### Task 3: Private deployment, live integration and handoff

**Files:** Create scripts/preview.mjs, integration tests, README.md; runtime artifacts outside tracked source.

**Interfaces:** Existing /api/training/progress, /api/health and /api/frames are read-only verification targets. Sites source helper returns exact project_id/commit/archive; native deployment returns URL.

- [ ] Add integration tests for remote queue→agent→isolated backend round trip and media, run red then implement any missing adapter and pass.
- [ ] Build and validate artifact (including migration journal), prepare source via supported Sites helper, privately deploy exact pushed commit, retain returned deployment/version identity.
- [ ] Start resource-bounded user service for bridge, inject service credential once in memory, verify live read-only health/progress/images through deployed Site.
- [ ] Browser QA on deployed private page, verify current training curves, controls present, connection indicator; no true write calls.
- [ ] Required fresh whole-source review, fix important findings with regression tests, run complete Site/agent suites and build validation.
- [ ] Confirm original training processes still advance; document URL, private login, local dependency and recovery. Commit/publish any reviewed changes before handoff.

## Execution ruling

User explicitly approved continuing implementation and private publication. Execute natively without another approval gate; no new long training or physical simulation is authorized by Site verification.
