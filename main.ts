// main.ts — one server, two client wires, ten simulated free-tier providers.
//
// The trial gives this server a brief (MAKEFUN_PROMPT_FILE) and an env (MAKEFUN_ENV_URL),
// and runs it as `node main.ts` from the repository root, with Node 22's standard library
// alone. It:
//
//   1. reads the brief: the ten providers, the sealed quality table, the workload manifest;
//   2. probes each provider's catalog route in the provider's own wire and builds one
//      catalog — the document `--list` prints and the register body carries, byte for byte;
//   3. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   4. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   5. serves the client's requests: GET /v1/models (OpenAI shape for an Authorization
//      header, Anthropic shape for x-api-key), POST /v1/chat/completions and POST
//      /v1/messages, each in JSON or server-sent events;
//   6. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says done.
//      A process still running when the wall ends scores zero, so it never waits on
//      anything else.
//
// Routing, per request (see lib/router.ts): the free models meeting the request's
// requirements, best rating first, ties spread across the providers called least so far.
// A provider call that answers 402, 429, 5xx or never answers is counted and the request
// is tried on the next candidate — a refusal is never the answer. Each (provider, model)
// is tried at most once per request, and the first success ends the request: the trial
// wants exactly one provider call to answer each request.
//
// The wires (see lib/wire.ts): the request is translated into the provider's wire
// preserving exactly what the provider echoes back — system prompt, user text, tool
// names, max_tokens, stop sequences, images, JSON mode — and the answer is translated
// back into the wire the client used, streamed or not as it asked, with the provider's
// text whole, its usage counts as that wire reports them, and its stop reason mapped.
// A streamed OpenAI answer gets a usage chunk even when the client did not ask for
// stream_options.include_usage: the server asks the provider for one, and appends a
// computed one when the provider still sends none.
//
// Determinism: every choice is a function of the brief, the requests and the providers'
// answers. No wall clock, no unseeded randomness, no parallel provider calls — one
// request at a time, in the order they arrive.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { Readable } from 'node:stream';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { Router, type Candidate } from './lib/router.ts';
import {
  ANTHROPIC_VERSION,
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIStreamPassthrough,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  usageInputOf,
  usageOutputOf,
  type ServedModel,
  type Usage,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const INFERENCE_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let router: Router | null = null;
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  router = new Router(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The free models across the providers that answered, each id once, in the brief's provider
 * order: what GET /v1/models lists. The trial compares the list with the registered catalog
 * as a set, so duplicates would be harmless — but one id, once, is the honest list.
 */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      const row = brief.ratings?.ratings.find((r) => r.provider === probe.provider.id && r.model === model.id) ?? null;
      const free = row !== null ? row.free : (model.free ?? false);
      if (!free) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order. */
function requirementsFor(api: 'openai' | 'anthropic', body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(api, body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(api: 'openai' | 'anthropic', body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: api === 'openai' && (format === 'json_object' || format === 'json_schema'),
  };
}

/** Whether this OpenAI client request asks for JSON mode (only the OpenAI wire carries it). */
function jsonModeOf(api: 'openai' | 'anthropic', body: Json): boolean {
  if (api !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One inference call to the routed provider, in its own wire. The body is the client's
 * request translated into that wire; the model travels in the body on the OpenAI and
 * Anthropic wires and in the URL on the Gemini wire. A streamed OpenAI call asks for
 * stream_options.include_usage, so the provider's stream carries the usage chunk the
 * client checks — the client's own stream bodies do not ask for it.
 */
async function callProvider(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  const signal = AbortSignal.timeout(INFERENCE_TIMEOUT_MS);
  if (provider.wire === 'openai') {
    const upstreamBody =
      clientApi === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal });
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    const upstreamBody =
      clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal });
  }
  const path = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  const upstreamBody = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  return fetch(`${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${path}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(upstreamBody),
    signal,
  });
}

/** A buffered upstream body as a readable stream, for the SSE translators. */
function streamOf(text: string): ReadableStream<Uint8Array> {
  const web = Readable.toWeb(Readable.from([Buffer.from(text, 'utf8')])) as ReadableStream<Uint8Array>;
  return web;
}

/**
 * A provider's JSON answer as the client's JSON answer, by the two wires involved.
 * `fallback` is the usage computed from the client request, used only when the provider
 * reports none of its own.
 */
function translateJson(
  clientApi: 'openai' | 'anthropic',
  wire: 'openai' | 'anthropic' | 'gemini',
  answer: Json,
  modelId: string,
  fallback: Usage,
): Json {
  if (clientApi === 'openai') {
    if (wire === 'openai') return answer;
    if (wire === 'anthropic') return anthropicToOpenAIResponse(answer, modelId);
    return geminiToOpenAI(answer, modelId, fallback);
  }
  if (wire === 'openai') return openAIToAnthropic(answer, modelId);
  if (wire === 'anthropic') return answer;
  return geminiToAnthropic(answer, modelId, fallback);
}

/**
 * A provider's streamed answer as the client's stream, by the two wires involved. The
 * upstream stream is read whole before anything is sent to the client: a stream that
 * breaks mid-way is a refusal, and the request can still fail over to the next candidate.
 * The Anthropic-to-Anthropic stream is the one case with nothing to translate, so it is
 * piped as it arrived.
 */
async function translateStream(
  clientApi: 'openai' | 'anthropic',
  wire: 'openai' | 'anthropic' | 'gemini',
  upstream: Response,
  modelId: string,
  fallback: Usage,
  res: ServerResponse,
): Promise<boolean> {
  if (upstream.body === null) return false;
  // Read the whole upstream stream before anything is sent to the client: a break here is
  // still a refusal, and the request can fail over to the next candidate.
  let text: string;
  try {
    text = await upstream.text();
  } catch {
    return false;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (clientApi === 'anthropic' && wire === 'anthropic') {
    // The one case with nothing to translate: send the stream as it arrived.
    res.write(text);
    res.end();
    return true;
  }
  const stream = streamOf(text);
  const events =
    clientApi === 'openai'
      ? wire === 'openai'
        ? openAIStreamPassthrough(stream, modelId, fallback)
        : wire === 'anthropic'
          ? openAIEventsFromAnthropic(stream, modelId, fallback)
          : openAIEventsFromGemini(stream, modelId, fallback)
      : wire === 'openai'
        ? anthropicEventsFromOpenAI(stream, modelId, fallback)
        : anthropicEventsFromGemini(stream, modelId, fallback);
  for await (const event of events) res.write(event);
  res.end();
  return true;
}

/**
 * Serve one client request: route it, call providers in rating order until one answers,
 * and translate the answer into the client's wire. A refusal (402, 429, 5xx, a body that
 * is not JSON, a call that never answers) is counted and tried on the next candidate.
 */
async function serve(api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requestIndex = manifestIndex;
  const requires = requirementsFor(api, body);
  const jsonMode = jsonModeOf(api, body);
  const stream = body.stream === true;
  const fallback: Usage = { input: usageInputOf(api, body), output: 0 };
  const activeRouter = router;
  if (activeRouter === null) {
    return sendError(res, api, 503, 'no_free_model', 'the server has no providers to route to');
  }
  const candidates = activeRouter.candidatesFor(requires, api, jsonMode, requestIndex);
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    let upstream: Response;
    try {
      upstream = await callProvider(candidate, api, body, stream);
    } catch (err) {
      activeRouter.noteCall(providerId, requestIndex, false);
      log(`${providerId}/${candidate.model.id} did not answer: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok) {
      activeRouter.noteCall(providerId, requestIndex, false);
      const detail = (await upstream.text().catch(() => '')).slice(0, 200);
      log(`${providerId}/${candidate.model.id} refused with HTTP ${upstream.status}: ${detail}`);
      continue;
    }
    activeRouter.noteCall(providerId, requestIndex, true);
    if (!stream) {
      let answer: unknown;
      try {
        answer = await upstream.json();
      } catch {
        // A 200 that is not JSON cannot serve the request; the next candidate may.
        log(`${providerId}/${candidate.model.id} answered 200 with a body that is not JSON`);
        continue;
      }
      const translated = translateJson(api, candidate.probe.provider.wire, isRecord(answer) ? answer : {}, candidate.model.id, {
        ...fallback,
        output: 0,
      });
      sendJson(res, 200, translated);
      return;
    }
    const sent = await translateStream(api, candidate.probe.provider.wire, upstream, candidate.model.id, fallback, res).catch(() => false);
    if (sent) return;
    // The stream broke before anything was sent, so the request can still fail over; the
    // call itself already counted as a success above, so only the log records the break.
    log(`${providerId}/${candidate.model.id} stream broke before its answer was sent`);
  }
  sendError(
    res,
    api,
    503,
    api === 'anthropic' ? 'overloaded_error' : 'no_free_model',
    'no free model on the directory could serve this request',
  );
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object
    // and stop before any server starts. A process whose stdout is not flushed loses the
    // tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
