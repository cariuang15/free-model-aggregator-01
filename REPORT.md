# free-model-aggregator-01 — build report

Bounty: **The best free AI model aggregator** — $500, 20 places (1st $100, 2nd $70,
3rd $50, 4th–10th $20, 11th–20th $14). Entries close **Mon 16 Nov 2026, 09:30 WIB**
(02:30 UTC). Leaderboard at build time: 6,396 agents, leader @xam-metri 99.0.

## What was built

One Node 22 server (`main.ts`, stdlib only) that serves the OpenAI Chat Completions
API (`POST /v1/chat/completions`) and the Anthropic Messages API
(`POST /v1/messages`) over ten simulated free-tier providers, in JSON and in
server-sent events, on all three upstream wires (OpenAI, Anthropic, Gemini).

- `main.ts` — the server: brief reading, catalog probe, register with the env,
  routes, rating-based routing with budget spreading, failover, `/work` polling.
- `lib/brief.ts` — the brief's three documents (provider directory, quality table,
  workload manifest), the catalog probe, and the catalog builder.
- `lib/wire.ts` — all six client↔provider wire translations (requests and answers,
  JSON and streaming), including the Gemini wire the starter lacked.
- `LICENSE` (MIT, with `SPDX-License-Identifier: MIT`), `README.md`,
  `package.json` (`{"type": "module"}`).

## How it works

- **Routing:** candidates are every free model with a quality-table row that meets
  the request's requirements, ordered by the table's rating minus a usage penalty
  per call the provider has already made (8 milli on the first attempt, 60 on a
  retry). Providers under their safe call cap sort first — the cap is the lowest
  per-trial budget the sealed table draws for that provider, so staying under it
  can never exhaust a provider. A JSON-mode request from an OpenAI client only
  goes to an openai- or gemini-wire provider, because the Anthropic wire cannot
  carry `response_format`.
- **Failover:** on a refusal the request moves to the next candidate. A
  "budget is spent" or "is out" message marks the provider dead for the trial;
  a "refused this call" is a scheduled failure and the provider stays usable.
  A retry orders fresh providers first: a provider's first call cannot fail
  (every failure schedule starts at call two), so a retry to a fresh provider
  ends the cascade instead of extending it.
- **Wire fidelity:** every request is forwarded in the provider's wire untouched
  in what the provider echoes — system prompt, user text with images, tools,
  stop sequences, `max_tokens`, JSON mode — and the provider's answer text,
  usage counts and stop reason come back whole, translated event by event for
  streams. Streaming OpenAI calls to an OpenAI provider set
  `stream_options.include_usage` so the usage chunk is present.
- **Determinism:** every choice derives from the brief, the requests and the
  providers' answers; no wall clock, no unseeded randomness. All three practice
  trials replayed byte for byte.

## Test results (grade-local, --driver process, all 3 practice instances)

| trial | score | quality | served | wire | failover | catalog | calls | working |
|-------|-------|---------|--------|------|----------|---------|-------|---------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 25    | 1 |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 25    | 1 |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 26    | 1 |

Mean score **99** (90% interval 99–99), 3 finished, 0 timeout, 0
nondeterministic. The house starter scores 38 on trial 0 (quality 38, served
40, wire 39, failover 0, catalog 100, 14 calls).

A stronger first-attempt usage penalty (20 milli/call) was tried and reverted:
it scored the same 99 but with lower quality (98 on trials 1–2) and no fewer
calls (24/25/28). The remaining ~5 extra calls per trial are the providers'
scheduled failures landing on call numbers the workload must use; they are
handled by failover, not avoidable from the brief.

`--list` verified: prints `{"providers":[]}` with no env set, and the full
catalog (10 providers, 28 models, free flags and context lengths) against a
live trial-0 env. The kit's `check` passes everything checkable locally
(competition_open, snapshot, license_osi, entrypoint, entry_size,
committed_provider_key); only `transcripts_present` fails, as expected before
bundling.

## How to run and test

```sh
# Grade all 3 practice trials locally (no Docker needed: Node 22 entry, process driver)
cd ~/workspace/makefun-aggregator-kit/free-model-aggregator-01-kit
node cli/makefun.mjs grade-local --kit . --bot ~/workspace/makefun-aggregator --driver process --instances 3

# Pre-submit checks (transcripts fail until the bundle exists)
node cli/makefun.mjs check --kit . --repo ~/workspace/makefun-aggregator

# Catalog document on its own (needs a trial env for the full catalog)
MAKEFUN_ENV_URL=http://127.0.0.1:18080 MAKEFUN_PROMPT_FILE=<trial-0 prompt.txt> node main.ts --list
```

## Submission steps still needed (parent / user)

1. Create a **new public GitHub repo** under `0x23466` (created after the
   competition opened, 14 Nov 2026 02:30 UTC), push this directory's files
   (`main.ts`, `lib/`, `LICENSE`, `README.md`, `package.json`, `REPORT.md`,
   `.gitignore`) — the kit lives in `~/workspace/makefun-aggregator-kit/` and
   must NOT be committed.
2. Bundle the transcripts of the sessions that built it with the kit CLI
   (`node cli/makefun.mjs bundle`, run from inside the unpacked kit, from the
   top-level agent once this subagent has finished) — the same facts as the
   chess entry: harness Muse, provider meta, model ipnext/avocado-5.17-v2
   (recorded), harness-version 1.3, effort default, MCP none.
3. Submit before **Mon 16 Nov 2026, 09:30 WIB**: repo URL + full commit SHA +
   transcript bundle. No entry fee, no wallet purchase needed; the entry is
   signed by a wallet on the @warisan account (escrowed prize). If @warisan's
   make.fun sign-in is via Solana wallet rather than GitHub 0x23466, add
   `.makefun/owner.json` with the full linked wallet address in a commit first
   (only the truncated `6YHj…mpV8` is known — the full address is needed).
4. If the transcript gate rejects the entry, its status names the failed
   checks and fixes; resubmission is allowed (max 10/day, 20 total).

## Honest limits

- The scored instances (drawn after entries close) have different provider
  budgets, failure schedules and outage providers than the practice trials.
  The routing adapts at runtime (it reads failures as they happen), but the
  practice score of 99 is not a guarantee for the scored instances — a bad
  draw (e.g. the outage provider being the one the workload leans on) can cost
  a few calls and, in the worst case, a request.
- The leader sits at 99.0, so this entry is most likely tied on score with
  the leaders; the tie-break is fewer provider calls, then earlier receipt.
  Mean 25.3 calls is competitive but not provably the lowest — the remaining
  extra calls are scheduled failures, and any entry serving all 20 requests
  pays some of them.
- The final is staged: every entry past the practice floor (0) runs the first
  scored instance once; the top 60 there run all 3, ranked on the mean. A
  single bad first instance can keep the entry out of the top 60.
- No model calls are made (a scripted program is a legal entry for this
  bounty); no money was spent building this, and nothing has been submitted.
