# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/0x23466/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/0x23466/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated free-tier model providers: it reads the trial's brief, probes every provider's catalog, picks the best free model each request can use, fails over when a provider refuses, and answers in the wire the client used — JSON and streaming, on all three upstream wires (OpenAI, Anthropic, Gemini).

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the catalog document the server registers, and exits 0.

## How it works

- **Routing:** candidates are every free model with a quality-table row that meets the request's requirements, ordered by the table's rating, best first, with providers under their safe call cap first (the lowest per-trial budget the sealed table draws, so staying under it can never exhaust a provider). A JSON-mode request from an OpenAI client only goes to an openai- or gemini-wire provider, because the Anthropic wire cannot carry `response_format`.
- **Failover:** on a refusal the request moves to the next candidate. A "budget is spent" or "is out" message marks the provider dead for the trial; a "refused this call" is a scheduled failure and the provider stays usable for later requests.
- **Wire fidelity:** every request is forwarded in the provider's wire untouched in what the provider echoes — system prompt, user text with images, tools, stop sequences, `max_tokens` and JSON mode — and the provider's answer text, usage counts and stop reason come back whole, translated event by event for streams.

---

<sub>Entered on [make.fun](https://make.fun/?ref=csc1bejp&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
