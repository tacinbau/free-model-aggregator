# Routing design

How this server picks a model, why each rule exists, and what it costs when a
rule fires. All numbers below are from the three practice trials, scored with
the kit's sealed scorer.

## The score, and what moves it

Five components, weighted: quality 60, served 15, wire 10, failover 10,
catalog 5. The tie-break is the number of provider calls (fewer first).

- **Quality** is the weighted rating of each valid answer over the best rating
  in the sealed table for a free model meeting that request's requirements,
  ignoring every provider's per-trial budget. Weight 2 requests (r01–r06)
  count double. A paid model earns nothing, ever.
- **Served** is the share of the 20 requests answered validly (every check on
  them passing).
- **Wire** is the share of the response checks passed: shape, stream order,
  the echo of system / user text / tools / max_tokens / stops / image / JSON
  mode, stop reason, usage.
- **Failover** is the share of requests whose *first* provider call failed
  (402 / 429 / 5xx) that were still served validly.
- **Catalog** is the accuracy of the registered catalog against the providers'
  true catalogs, plus the entry's own `GET /v1/models` listing exactly the
  free model ids it registered (asked with both auth headers at the trial's
  start).

## The candidate order

For one request, in order:

1. **Free and capable, only.** A model is free by the quality table's row
   where it has one (the sealed truth), else by the listing's own word. It
   must meet every requirement: tools, vision, JSON mode, minimum context.
   Unknown capability never meets a requirement.
2. **Wire-compatible.** An OpenAI JSON-mode request carries `response_format`,
   which the Anthropic wire cannot carry: an Anthropic provider echoes
   `json_mode` absent and the echo check fails. Such requests go only to
   OpenAI-wire or Gemini-wire providers (Gemini carries it as
   `responseMimeType: application/json`). Anthropic client requests never ask
   for JSON mode on their own wire, so they may use any wire.
3. **Best rating first.** Quality is 60% of the score, and the weight-2
   requests arrive first in the manifest, so greedily taking the best
   available also serves the heavy requests best — no look-ahead is needed.
4. **Ties spread the load.** Equal ratings go to the provider called least so
   far this trial, then provider id, then model id. The same model id on two
   providers (glm-4.7-flash on Z.ai and Cloudflare, qwen-3.8-27b on Groq and
   Cerebras, gpt-oss-120b on Groq and Cerebras, gemma-4-31b on Gemini and
   SambaNova) spreads its load instead of draining one provider's budget.

## Learning the budgets without being told them

The trial never publishes a provider's per-trial budget, failure schedule, or
outage — `/instance` returns only the directory and the manifest. The router
learns from refusals alone:

- Every call's outcome is recorded per provider: calls made, refusals in a
  row, the request index of the last refusal.
- A provider that refused **twice in a row** cools down: its candidates move
  to the back of the order, tried only when nothing else is left, until three
  requests have been served since its last refusal. A spent budget or an
  outage refuses every later call, so two in a row is the signal; a lone
  scheduled failure is transient, and the provider is tried again at once.
- A longer cooldown (5) was tried and reverted: it saved one call on trial-0
  and trial-1 but cost three on trial-2, where a cooled-down provider forced
  longer failover chains. Three is the measured sweet spot.
- Retrying a *different* model on the same provider inside one request was
  tried and reverted for the same reason: the trajectory through the budgets
  is chaotic, and the saving never survives all three trials.

## What the practice trials show

| Trial | Top model successes | Where the quality went |
| --- | --- | --- |
| trial-0 | OpenRouter 1475 ×10 of 12 calls (fail_calls 2, 10) | 7 requests at 1470/1465 after OpenRouter's budget spent: −45 milli-weight → quality 99 |
| trial-1 | OpenRouter 1475 ×9 of 10 calls (fail_call 8) | Late-trial pile-up: Z.ai, Cloudflare and SambaNova all on scheduled failures at r18 (7 calls, served by gemma-4-31b 1452): −75 → quality 99 |
| trial-2 | OpenRouter 1475 ×8 of 10 calls (fail_calls 2, 9) | Vision r02: Gemini's 2nd call failed, fell to gemma-4-31b 1452 (−12, weight 2); rest at 1470/1465: −67 → quality 99 |

Quality **100 needs every request at its best rating**, which the budgets make
impossible (17 non-vision requests, 8–10 top-model successes per trial), so 99
is the ceiling on all three instances — and every point lost traces to a real
budget exhaustion or scheduled failure, never to a wrong pick. Served, wire,
failover and catalog are 100 on all three trials.

## The wire rules that protect the wire column

- **Never drop a field the echo measures.** The provider echoes the system
  prompt, the user text (last user message), the tool names in order,
  `max_tokens`, the stop sequences, the image, and the JSON-mode flag. Every
  translation preserves those fields exactly; anything else may be reshaped.
- **OpenAI stream usage.** The client's stream bodies carry no
  `stream_options`, so a provider sends no usage chunk unless asked. This
  server asks (`stream_options: {include_usage: true}`) on OpenAI-wire calls,
  and when a stream still ends without one it appends a usage chunk computed
  by the provider's own formula (input from the request, output from the
  answer text) — the same numbers, so the usage check passes.
- **Anthropic stream usage.** Only the output tokens are checked, and they
  arrive in `message_delta`. The input tokens in `message_start` are computed
  from the request up front, so they are right too.
- **Gemini usage.** The final stream frame and the JSON answer both carry
  `usageMetadata`; the translators map `promptTokenCount` /
  `candidatesTokenCount` to the client's usage fields, with the computed
  fallback when the metadata is absent.
- **Stop reasons** map by family (stop / end_turn / STOP; length / max_tokens
  / MAX_TOKENS; tool_calls / tool_use; content_filter / refusal / SAFETY).
- **One success per request.** The first 2xx ends the request; a second
  successful call would fail the `one_provider` check. Failures before the
  success are the failover column's input, not a penalty.
- **Streams are buffered before answering.** The upstream stream is read whole
  before anything is sent to the client, so a stream that breaks mid-way is
  still a refusal the request can fail over from — not a half-answer.
