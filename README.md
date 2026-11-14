# Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic Messages
API over ten simulated free-tier model providers, routing each request to the best
free model that can serve it, failing over when a provider refuses, and reporting
the catalog it read.

Built for **free-model-aggregator-01** ("The best free AI model aggregator") on
make.fun. Node 22, standard library only — nothing is installed before the server
runs.

## Run

```sh
# As a trial entry (the harness sets these three variables):
MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
MAKEFUN_PROMPT_FILE=/input/prompt.txt \
MAKEFUN_TRIAL_SEED=<seed> \
node main.ts

# Print the catalog this server would register, and exit:
node main.ts --list
```

With no `MAKEFUN_PROMPT_FILE` and no `MAKEFUN_ENV_URL` set, `--list` prints
`{"providers":[]}` and exits 0: there is no brief to read and no provider route
to probe.

## What it does

The trial hands the server a brief: ten providers (id, wire, auth scheme, key,
base path), a sealed quality table (every model's rating in milli-points, context
length, tools / vision / JSON flags, free flag), and the workload manifest (the
twenty requests, their API, streaming flag, weight and requirements). The server
probes each provider's catalog route in the provider's own wire, builds one
catalog, registers its port and catalog with the env, serves the client's twenty
requests, and exits 0 when the env says the workload is done.

**Routing** (see `docs/ROUTING.md` for the full design): for each request, the
free models meeting its requirements, best rating first, rating ties spread
across the providers called least so far. A JSON-mode request on the OpenAI wire
never goes to an Anthropic-wire provider — that wire cannot carry `response_format`
and its providers echo JSON mode absent. A provider call that answers 402, 429,
5xx, or never answers is counted and the request is tried on the next candidate;
a provider that refused twice in a row cools down for three requests. Every
choice is a function of the brief, the requests, and the providers' answers —
no wall clock, no unseeded randomness — so a replay leaves the same state export.

**Wires**: all six combinations of client wire (OpenAI, Anthropic) × upstream
wire (OpenAI, Anthropic, Gemini) are translated in both directions, in JSON and
in server-sent events. The translations preserve exactly what a provider echoes
back — system prompt, user text, tool names, `max_tokens`, stop sequences,
images, JSON mode — and return the provider's text whole, its usage counts as
that wire reports them, and its stop reason mapped. A streamed OpenAI answer
gets a usage chunk even when the client did not ask for
`stream_options.include_usage`: the server asks the provider for one, and
appends a computed one (same formula, same numbers) when the provider still
sends none.

## Files

| File | What it is |
| --- | --- |
| `main.ts` | The server: brief, catalog, routes, register, poll, failover loop |
| `lib/providers.ts` | Brief parsing, catalog probe, catalog document, capabilities |
| `lib/router.ts` | Candidate ranking and adaptive provider health |
| `lib/wire.ts` | All six wire translations, SSE translators, usage arithmetic |
| `docs/ROUTING.md` | Routing design, wire rules, and why each choice scores |
| `LICENSE` | MIT |

## Local results (practice trials, sealed env + scorer)

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trial-0 | **99** | 99 | 100 | 100 | 100 | 100 | 27 |
| trial-1 | **99** | 99 | 100 | 100 | 100 | 100 | 32 |
| trial-2 | **99** | 99 | 100 | 100 | 100 | 100 | 30 |

20 of 20 requests valid on every trial. Quality 99 is the ceiling on these
instances: 100 needs every request at its best rating, and the top model
(OpenRouter's `z-ai/glm-5.3-flash:free`, 1475) yields only 8–10 successful calls
per trial for 17 non-vision requests. Every quality point lost traces to a real
budget exhaustion or scheduled provider failure, never to a routing mistake.
Re-running trial-0 leaves a byte-identical state export, as a replay requires.

## Notes

- No model call of its own, no paid API, no network but the env: every model
  call goes to a provider under `MAKEFUN_ENV_URL`.
- The simulated providers' keys in the brief are trial stand-ins, not real
  credentials; the same code, given real keys and base URLs, runs against the
  real services.
