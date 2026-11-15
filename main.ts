// free-model-aggregator-01: one server, two client wires, ten simulated providers.
//
// This server reads the trial's brief, probes every provider's catalog route, builds one
// catalog, registers with the env, and serves the client's 20 requests: POST
// /v1/chat/completions (OpenAI) and POST /v1/messages (Anthropic), in JSON and in
// server-sent events. Every request is answered by a free model that meets its
// requirements, on the provider's own wire (openai, anthropic or gemini), and the answer
// comes back in the wire the client used, with the provider's text, usage and stop reason
// carried through whole.
//
// ROUTING (quality is 60% of the score, served 15%, failover 10%)
//   Candidates are every free model with a quality-table row that meets the request's
//   requirements, on a wire that can carry it: a JSON-mode request from an OpenAI client
//   only goes to an openai- or gemini-wire provider, because the Anthropic wire cannot
//   carry response_format. They are ordered by the table's rating, best first, with
//   providers under their safe call cap first: a provider's per-trial budget is drawn
//   between its sealed minimum and maximum, so the minimum is a cap no trial can go
//   under, and staying under it can never exhaust a provider. On a refusal the request
//   fails over to the next candidate: a "budget is spent" or "is out" message marks the
//   provider dead for the trial (its budget is gone or its outage began), a "refused this
//   call" is a scheduled failure and the provider stays usable for later requests.
//
// WIRE FIDELITY (wire is 10% of the score; a failed echo check voids the request)
//   Every request is forwarded in the provider's wire untouched in what the provider
//   echoes: system prompt, user text with images, tools, stop sequences, max_tokens and
//   JSON mode. Streaming answers are translated event by event, so the concatenated text
//   is the provider's whole answer and the usage chunk carries the provider's counts.
//
// DETERMINISM
//   Every choice derives from the brief, the requests and the providers' answers: no
//   wall clock and no unseeded randomness, so a replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicStreamToOpenAI,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIJson,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiToAnthropicJson,
  geminiToOpenAIJson,
  openAIError,
  openAIModelList,
  openAIStreamToAnthropic,
  openAIToAnthropic,
  openAIToAnthropicJson,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

/**
 * The safe call cap per provider: the lowest per-trial budget the sealed table draws for
 * it, so a provider called at most this often can never run out mid-trial. An unknown
 * provider gets 4. The caps total far more than the 20 requests plus failover retries.
 */
const SAFE_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Inference calls made to each provider so far, failed ones included: the env counts them the same way. */
const callsMade = new Map<string, number>();
/** Providers known spent or out for this trial: never called again. */
const dead = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists, deduped by id. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  providerIndex: number;
}

/**
 * Every free model that can serve one request, best rated first. A model counts only with a
 * quality-table row: the scorer rates answers by that table, and a model it does not know is
 * never a legal pick. A JSON-mode request from an OpenAI client only goes to a wire that can
 * carry response_format (openai or gemini); an Anthropic client carries no response_format,
 * so any wire serves its JSON requests as long as the model itself takes JSON.
 *
 * The order is the rating minus a usage penalty per call the provider has already made: 8
 * milli on the first attempt, which spreads the load over providers before any of them is
 * deep into its failure schedule, and 60 milli on a retry, which puts a provider that has
 * not been called yet first — its first call cannot fail (every failure schedule starts
 * at call two), so a retry to a fresh provider ends the cascade instead of extending it.
 * Providers over their safe cap always sort after providers under it.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean, attempt: number): Candidate[] {
  const out: Candidate[] = [];
  probes.forEach((probe, providerIndex) => {
    if (!probe.ok) return;
    if (dead.has(probe.provider.id)) return;
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') return;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (row === null || !row.free) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row.rating_milli, providerIndex });
    }
  });
  const penalty = attempt === 0 ? 8 : 60;
  const underCap = (c: Candidate): number => ((callsMade.get(c.probe.provider.id) ?? 0) < (SAFE_CAP[c.probe.provider.id] ?? 4) ? 0 : 1);
  const used = (c: Candidate): number => callsMade.get(c.probe.provider.id) ?? 0;
  const effective = (c: Candidate): number => c.rating - penalty * used(c);
  out.sort(
    (a, b) =>
      underCap(a) - underCap(b) ||
      effective(b) - effective(a) ||
      used(a) - used(b) ||
      a.providerIndex - b.providerIndex ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether an OpenAI client request asks for JSON mode. */
function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
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

/** The error message of a provider's refusal, in whichever wire's envelope it came. */
function errorMessageOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value)) {
      const error = isRecord(value.error) ? value.error : null;
      if (error !== null && typeof error.message === 'string') return error.message;
      if (typeof value.message === 'string') return value.message;
    }
  } catch {
    // Not JSON: the text itself is the message.
  }
  return text;
}

/**
 * What a refusal says about the provider: 'dead' when its budget is spent, its outage began,
 * or the call was refused as invalid (a 400/401 is this server's own error, never retried on
 * the same provider); 'transient' for a scheduled failure or anything else, which leaves the
 * provider usable for later requests.
 */
function refusalKind(status: number, message: string): 'dead' | 'transient' {
  if (status === 400 || status === 401) return 'dead';
  if (message.includes('budget is spent')) return 'dead';
  if (message.includes('is out')) return 'dead';
  return 'transient';
}

/** One upstream call: the request in the provider's wire, its answer, or the refusal. */
interface UpstreamCall {
  ok: boolean;
  status: number;
  response: Response | null;
  error: string | null;
}

async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<UpstreamCall> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = body.stream === true;
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstreamBody = { ...upstreamBody, stream: true, stream_options: { include_usage: true } };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
  } else {
    const built = clientApi === 'openai' ? openAIToGemini(body, candidate.model.id) : anthropicToGemini(body, candidate.model.id);
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    const query = stream ? '?alt=sse' : '';
    url = `${base}/v1beta/models/${encodeURIComponent(built.urlModel)}:${action}${query}`;
    upstreamBody = built.body;
  }
  if (stream) headers.accept = 'text/event-stream';
  callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      const message = errorMessageOf(text.slice(0, 2000));
      if (refusalKind(response.status, message) === 'dead') dead.add(provider.id);
      return { ok: false, status: response.status, response: null, error: `${provider.id} answered HTTP ${response.status}: ${message}` };
    }
    return { ok: true, status: response.status, response, error: null };
  } catch (err) {
    return { ok: false, status: 0, response: null, error: `${provider.id} could not be reached: ${(err as Error).message}` };
  }
}

/** A provider's JSON answer as the client's JSON answer, in the client's wire. */
function convertJson(clientApi: 'openai' | 'anthropic', wire: 'openai' | 'anthropic' | 'gemini', answer: Json, model: string): Json {
  if (clientApi === 'openai') {
    if (wire === 'openai') return answer;
    if (wire === 'anthropic') return anthropicToOpenAIJson(answer, model);
    return geminiToOpenAIJson(answer, model);
  }
  if (wire === 'openai') return openAIToAnthropicJson(answer, model);
  if (wire === 'anthropic') return answer;
  return geminiToAnthropicJson(answer, model);
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write a stream of SSE event strings to the client. */
async function writeEvents(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  try {
    for await (const event of events) res.write(event);
  } finally {
    res.end();
  }
}

/**
 * One client request, served by the first candidate that answers: tries each candidate in
 * order, failing over on every refusal, and returns the first answer translated into the
 * client's wire. A streamed answer is translated event by event; a JSON answer is parsed
 * before anything is sent, so a broken answer fails over instead of reaching the client.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const requires = requirementsFor(body);
  const jsonMode = jsonModeOf(clientApi, body);
  const stream = body.stream === true;
  let lastError: string | null = null;
  // The candidate order is recomputed before every attempt: each attempt changes the
  // providers' call counts, and a retry orders fresh providers first.
  const tried = new Set<string>();
  for (let attempt = 0; ; attempt += 1) {
    const candidates = candidatesFor(requires, clientApi, jsonMode, attempt).filter((c) => !tried.has(`${c.probe.provider.id}/${c.model.id}`));
    if (candidates.length === 0) break;
    const candidate = candidates[0];
    tried.add(`${candidate.probe.provider.id}/${candidate.model.id}`);
    const call = await callUpstream(candidate, clientApi, body);
    if (!call.ok || call.response === null) {
      lastError = call.error;
      log(`failover: ${call.error ?? 'no answer'}`);
      continue;
    }
    const wire = candidate.probe.provider.wire;
    const model = candidate.model.id;
    try {
      if (!stream) {
        const answer: unknown = await call.response.json();
        if (!isRecord(answer)) throw new Error('the provider answered a body that is not one JSON object');
        sendJson(res, 200, convertJson(clientApi, wire, answer, model));
        return;
      }
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (clientApi === 'openai' && wire === 'openai') {
        if (call.response.body === null) throw new Error('the provider answered a stream with no body');
        await pipeStream(call.response.body, res);
      } else if (clientApi === 'anthropic' && wire === 'anthropic') {
        if (call.response.body === null) throw new Error('the provider answered a stream with no body');
        await pipeStream(call.response.body, res);
      } else if (clientApi === 'openai' && wire === 'anthropic') {
        if (call.response.body === null) throw new Error('the provider answered a stream with no body');
        await writeEvents(anthropicStreamToOpenAI(call.response.body, model), res);
      } else if (clientApi === 'anthropic' && wire === 'openai') {
        if (call.response.body === null) throw new Error('the provider answered a stream with no body');
        await writeEvents(openAIStreamToAnthropic(call.response.body, model), res);
      } else if (clientApi === 'openai' && wire === 'gemini') {
        if (call.response.body === null) throw new Error('the provider answered a stream with no body');
        await writeEvents(geminiStreamToOpenAI(call.response.body, model), res);
      } else {
        if (call.response.body === null) throw new Error('the provider answered a stream with no body');
        await writeEvents(geminiStreamToAnthropic(call.response.body, model), res);
      }
      return;
    } catch (err) {
      // The answer could not be read or translated: if nothing was sent yet, fail over; a
      // stream already in flight cannot be taken back, so the request ends there.
      if (!res.headersSent) {
        lastError = `${candidate.probe.provider.id} answered unreadably: ${(err as Error).message}`;
        log(`failover: ${lastError}`);
        continue;
      }
      log(`stream failed mid-answer: ${(err as Error).message}`);
      return;
    }
  }
  log(`unserved: ${lastError ?? 'no free model on the directory meets this request'}`);
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'openai', body);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'anthropic', body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
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
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
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
