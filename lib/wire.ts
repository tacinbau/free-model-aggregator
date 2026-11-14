// wire.ts — every translation between the two client wires and the three upstream wires.
//
// The client speaks OpenAI Chat Completions or Anthropic Messages. A provider speaks OpenAI,
// Anthropic or Gemini. That is six combinations, and each must be exact: the provider echoes
// back what it parsed — the system prompt, the user text, the tool names, max_tokens, the stop
// sequences, the image, the JSON-mode flag — and the client compares that echo with what it
// sent. A field rewritten or dropped loses the request. So every translation below preserves
// exactly the fields the echo measures, and drops nothing else the provider's parser reads.
//
// The echo rules, per wire (from the env's parsers):
//   openai    system = system messages' text joined "\n"; user = the LAST user message's
//             text; tools = tools[].function.name in order; max_tokens (or
//             max_completion_tokens); stop = body.stop; image = the last image_url in a user
//             message; json_mode = response_format.type is json_object or json_schema.
//   anthropic system = body.system text; user = the LAST user message's text blocks; tools =
//             tools[].name in order; max_tokens; stop = body.stop_sequences; image = the
//             last image block in a user message; json_mode is always false — the Anthropic
//             wire cannot carry it, so an OpenAI JSON-mode request never goes to an
//             Anthropic-wire provider (see router.ts).
//   gemini    system = systemInstruction parts text; user = the LAST non-model content's
//             parts text; tools = tools[].functionDeclarations[].name in order;
//             max_tokens = generationConfig.maxOutputTokens; stop =
//             generationConfig.stopSequences; image = the last inlineData in a user content;
//             json_mode = generationConfig.responseMimeType is application/json.
//
// Usage is the provider's own arithmetic, and the client checks it: input tokens are
// 1 + floor((system + user + tools-joined-by-comma) / 4), output tokens are
// 1 + floor(answer text / 4). A streamed OpenAI answer carries its usage only in a final
// usage chunk, which the provider sends only when the request asked for it with
// stream_options.include_usage — the client's stream bodies do not ask, so this server asks
// on their behalf, and when a stream still ends without a usage chunk it appends one computed
// from the request and the answer text (the same numbers, by the same formula).
//
// Stop reasons map by family: stop / end_turn / STOP are one family, length / max_tokens /
// MAX_TOKENS another, tool_calls / tool_use a third, content_filter / refusal / SAFETY a
// fourth. A provider here always answers with the first family, but the maps are complete.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the
// same body and the same provider answer give the same bytes on a replay. Model lists use
// fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The Anthropic version header the Messages wire wants. */
export const ANTHROPIC_VERSION = '2023-06-01';

/** Usage as the provider counts it: input from the request, output from the answer text. */
export interface Usage {
  input: number;
  output: number;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** The text of the system messages in an OpenAI body, joined the way the provider joins them. */
function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'system') continue;
    const text = textOf(message.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** The user text of the LAST user message in an OpenAI body, as the provider reads it. */
function openAIUserText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let userText = '';
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'user') continue;
    userText = textOf(message.content);
  }
  return userText;
}

/** The tool names of an OpenAI body, in order. */
function openAIToolNames(body: Json): string[] {
  const out: string[] = [];
  if (!Array.isArray(body.tools)) return out;
  for (const tool of body.tools) {
    if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') out.push(tool.function.name);
  }
  return out;
}

/** The user text of the LAST user message in an Anthropic body, as the provider reads it. */
function anthropicUserText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let userText = '';
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'user') continue;
    userText = textOf(message.content);
  }
  return userText;
}

/** The tool names of an Anthropic body, in order. */
function anthropicToolNames(body: Json): string[] {
  const out: string[] = [];
  if (!Array.isArray(body.tools)) return out;
  for (const tool of body.tools) {
    if (isRecord(tool) && typeof tool.name === 'string') out.push(tool.name);
  }
  return out;
}

/**
 * The provider's input-token count for a client request, by its own formula:
 * 1 + floor((system text + user text + tool names joined by ",") / 4). The translation is
 * faithful, so the provider parses the same three strings out of the translated body.
 */
export function usageInputOf(api: 'openai' | 'anthropic', body: Json): number {
  const systemText = api === 'openai' ? openAISystemText(body) : textOf(body.system);
  const userText = api === 'openai' ? openAIUserText(body) : anthropicUserText(body);
  const tools = api === 'openai' ? openAIToolNames(body) : anthropicToolNames(body);
  const prompt = systemText.length + userText.length + tools.join(',').length;
  return 1 + Math.floor(prompt / 4);
}

/** The provider's output-token count for an answer text: 1 + floor(length / 4). */
export function usageOutputOf(text: string): number {
  return 1 + Math.floor(text.length / 4);
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Request translations: client body -> upstream body
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an inline image. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the
 * system prompt becomes a system message, image blocks become image_url data URLs, tool_use
 * blocks become tool_calls, tool_result blocks become tool messages, tools and tool_choice
 * are rewritten, and the output cap, temperature, top_p, stop sequences and streaming pass
 * through. The provider echoes the same system, user text, tools, max_tokens and stops.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block: text or an image. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: the
 * system messages become the system string, user parts become Anthropic blocks, tool_calls
 * become tool_use blocks, tool messages become tool_result blocks, and max_tokens, stop,
 * temperature and top_p pass through. JSON mode is NOT carried — the Anthropic wire has no
 * field for it — so the router never sends a JSON-mode request down this path.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const system = openAISystemText(body);
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = blockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini content part: text or inline image data. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini content part: text or inline image data. */
function geminiPartOfAnthropic(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return {
        inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data },
      };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** The stop sequences of an OpenAI body, as a list (a lone string becomes one entry). */
function openAIStops(body: Json): string[] {
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  return [];
}

/** generationConfig shared by both Gemini translations: caps, stops, JSON mode, sampling. */
function geminiConfig(opts: {
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: unknown;
  topP: unknown;
}): Json {
  const config: Json = {};
  if (opts.maxTokens !== null) config.maxOutputTokens = opts.maxTokens;
  if (opts.stops.length > 0) config.stopSequences = opts.stops;
  if (opts.jsonMode) config.responseMimeType = 'application/json';
  if (typeof opts.temperature === 'number') config.temperature = opts.temperature;
  if (typeof opts.topP === 'number') config.topP = opts.topP;
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: the system messages
 * become systemInstruction, user parts become user contents, assistant text becomes model
 * contents, tools become functionDeclarations, and max_tokens, stop, JSON mode, temperature
 * and top_p land in generationConfig. Tool messages carry nothing the echo measures and are
 * left out: mapping one as a user content would overwrite the user text the provider echoes.
 * The model itself travels in the URL, not in this body.
 */
export function openAIToGemini(body: Json): Json {
  const system = openAISystemText(body);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'tool') continue;
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      if (text !== '') contents.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const geminiPart = geminiPartOf(part);
      if (geminiPart !== null) parts.push(geminiPart);
    }
    contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  out.generationConfig = geminiConfig({
    maxTokens,
    stops: openAIStops(body),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: body.temperature,
    topP: body.top_p,
  });
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: system becomes
 * systemInstruction, user blocks become user contents (tool_result blocks carry no text the
 * provider echoes, so a message holding only one still lands as a user content with an
 * empty text part, keeping the last-user-text echo honest), assistant text becomes model
 * contents, and tools, max_tokens, stop sequences and sampling carry over. Anthropic never
 * asks for JSON mode, so no responseMimeType is set.
 */
export function anthropicToGemini(body: Json): Json {
  const system = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      if (text !== '') contents.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block) || block.type === 'tool_result') continue;
      const geminiPart = geminiPartOfAnthropic(block);
      if (geminiPart !== null) parts.push(geminiPart);
    }
    contents.push({ role: 'user', parts: parts.length > 0 ? parts : [{ text: '' }] });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          input_schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  out.generationConfig = geminiConfig({
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    stops,
    jsonMode: false,
    temperature: body.temperature,
    topP: body.top_p,
  });
  return out;
}

// ---------------------------------------------------------------------------
// Response translations: upstream answer -> client answer (JSON, non-streaming)
// ---------------------------------------------------------------------------

/** A Chat Completions answer as an Anthropic Messages object. `model` is named back. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: openAIStopToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** The text of a Gemini candidate: its parts' text joined, as the client reads it. */
function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini answer's finishReason, or null. */
function geminiFinish(body: Json): unknown {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  return candidate.finishReason ?? null;
}

/** A Gemini answer's usage, or the fallback computed from the client request and the text. */
function geminiUsage(body: Json, fallback: Usage): { prompt: number; completion: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const prompt = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : fallback.input;
  const completion = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : fallback.output;
  return { prompt, completion };
}

/** A Gemini answer as an OpenAI Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string, fallback: Usage): Json {
  const text = geminiText(body);
  const usage = geminiUsage(body, { ...fallback, output: usageOutputOf(text) });
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: geminiStopToOpenAI(geminiFinish(body)), logprobs: null }],
    usage: { prompt_tokens: usage.prompt, completion_tokens: usage.completion, total_tokens: usage.prompt + usage.completion },
  };
}

/** A Gemini answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string, fallback: Usage): Json {
  const text = geminiText(body);
  const usage = geminiUsage(body, { ...fallback, output: usageOutputOf(text) });
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiStopToAnthropic(geminiFinish(body)),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt, output_tokens: usage.completion },
  };
}

/** An Anthropic answer as an OpenAI Chat Completions object. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const texts: string[] = [];
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    if (block.type === 'tool_use') {
      calls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const text = texts.join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

// ---------------------------------------------------------------------------
// Server-sent events
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (null for a bare data frame) and its data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a readable stream into SSE frames, decoding incrementally. */
export async function* readSseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield parseFrame(frame);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/** Copy a provider's event stream to the client as it arrives, byte for byte. */
export async function pipeStream(stream: ReadableStream<Uint8Array>, res: { write: (chunk: string | Buffer) => void; end: () => void }): Promise<void> {
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

/** An OpenAI chunk: one delta (or the finish), on the shared completion id. */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

/** An OpenAI usage chunk: no choices, just the token counts the client checks. */
function openAIUsageChunk(model: string, usage: Usage): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/**
 * An OpenAI event stream as an OpenAI event stream: every chunk forwarded as it arrived,
 * with a usage chunk appended when the provider sent none (it only sends one when the
 * request asked for stream_options.include_usage), and [DONE] to close. `fallback` is the
 * usage computed from the client request and the streamed text.
 */
export async function* openAIStreamPassthrough(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  fallback: Usage,
): AsyncGenerator<string> {
  let sawUsage = false;
  let sawDone = false;
  let text = '';
  for await (const frame of readSseFrames(upstream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      sawDone = true;
      yield 'data: [DONE]\n\n';
      break;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) sawUsage = true;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice !== null && isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
    yield `data: ${JSON.stringify(chunk)}\n\n`;
  }
  if (!sawUsage) yield `data: ${JSON.stringify(openAIUsageChunk(model, { input: fallback.input, output: usageOutputOf(text) }))}\n\n`;
  if (!sawDone) yield 'data: [DONE]\n\n';
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first
 * chunk, content_block_start/delta/stop per text block or tool call, then message_delta with
 * the stop reason and the usage, then message_stop. `fallback` carries the input tokens the
 * request implies (message_start needs them before any chunk arrives) and stands in for
 * output tokens when the provider sent no usage chunk.
 */
export async function* anthropicEventsFromOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  fallback: Usage,
): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = fallback.input;
  let outputTokens: number | null = null;
  let text = '';

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens ?? usageOutputOf(text) },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of readSseFrames(upstream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      for (const event of [...startEvents(), ...finishEvents(pendingStop ?? 'end_turn')]) yield event;
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    for (const event of startEvents()) yield event;
    if (typeof delta.content === 'string' && delta.content !== '') {
      text += delta.content;
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: typeof fn.name === 'string' ? fn.name : '',
            input: {},
          },
        });
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } });
      }
    }
    // Hold the stop reason until [DONE]: OpenAI sends the usage after the finish_reason, and
    // Anthropic's message_delta carries the output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = openAIStopToAnthropic(choice.finish_reason);
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/**
 * An Anthropic event stream as an OpenAI event stream: the role chunk, the text deltas, a
 * finish chunk with the mapped stop reason, a usage chunk, and [DONE]. The Anthropic stream
 * always carries its usage (input in message_start, output in message_delta); the fallback
 * stands in only if one is missing.
 */
export async function* openAIEventsFromAnthropic(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  fallback: Usage,
): AsyncGenerator<string> {
  let sawRole = false;
  let sawFinish = false;
  let sawDone = false;
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  let stopReason: string | null = null;
  let text = '';
  for await (const frame of readSseFrames(upstream)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
      if (!sawRole) {
        sawRole = true;
        yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
      }
    } else if (frame.event === 'content_block_delta') {
      if (isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        text += data.delta.text;
        if (!sawRole) {
          sawRole = true;
          yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
        }
        yield `data: ${JSON.stringify(openAIChunk(model, { content: data.delta.text }, null))}\n\n`;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = anthropicStopToOpenAI(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      break;
    }
  }
  if (!sawRole) yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
  if (!sawFinish) {
    sawFinish = true;
    yield `data: ${JSON.stringify(openAIChunk(model, {}, stopReason ?? 'stop'))}\n\n`;
  }
  yield `data: ${JSON.stringify(
    openAIUsageChunk(model, { input: inputTokens ?? fallback.input, output: outputTokens ?? usageOutputOf(text) }),
  )}\n\n`;
  if (!sawDone) yield 'data: [DONE]\n\n';
}

/**
 * A Gemini event stream as an OpenAI event stream: the role chunk, one content chunk per
 * text piece, a finish chunk, a usage chunk from the final frame's usageMetadata (or the
 * computed fallback), and [DONE].
 */
export async function* openAIEventsFromGemini(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  fallback: Usage,
): AsyncGenerator<string> {
  let sawRole = false;
  let sawFinish = false;
  let text = '';
  let usage: Usage | null = null;
  let finish: string | null = null;
  for await (const frame of readSseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const parts = isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
    for (const part of parts) {
      if (!isRecord(part) || typeof part.text !== 'string' || part.text === '') continue;
      text += part.text;
      if (!sawRole) {
        sawRole = true;
        yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
      }
      yield `data: ${JSON.stringify(openAIChunk(model, { content: part.text }, null))}\n\n`;
    }
    if (typeof candidate.finishReason === 'string') finish = geminiStopToOpenAI(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      usage = {
        input: typeof data.usageMetadata.promptTokenCount === 'number' ? data.usageMetadata.promptTokenCount : fallback.input,
        output: typeof data.usageMetadata.candidatesTokenCount === 'number' ? data.usageMetadata.candidatesTokenCount : usageOutputOf(text),
      };
    }
  }
  if (!sawRole) yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
  if (!sawFinish) {
    sawFinish = true;
    yield `data: ${JSON.stringify(openAIChunk(model, {}, finish ?? 'stop'))}\n\n`;
  }
  yield `data: ${JSON.stringify(openAIUsageChunk(model, usage ?? { input: fallback.input, output: usageOutputOf(text) }))}\n\n`;
  yield 'data: [DONE]\n\n';
}

/**
 * A Gemini event stream as the Anthropic event sequence: message_start (the input tokens
 * are computed from the request — the usageMetadata only arrives in the last frame),
 * content_block_start/delta/stop, message_delta with the stop reason and the output
 * tokens, then message_stop.
 */
export async function* anthropicEventsFromGemini(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  fallback: Usage,
): AsyncGenerator<string> {
  let started = false;
  let text = '';
  let usage: Usage | null = null;
  let finish: string | null = null;
  yield sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_stream',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: fallback.input, output_tokens: 0 },
    },
  });
  started = true;
  yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for await (const frame of readSseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const parts = isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
    for (const part of parts) {
      if (!isRecord(part) || typeof part.text !== 'string' || part.text === '') continue;
      text += part.text;
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: part.text } });
    }
    if (typeof candidate.finishReason === 'string') finish = geminiStopToAnthropic(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      usage = {
        input: typeof data.usageMetadata.promptTokenCount === 'number' ? data.usageMetadata.promptTokenCount : fallback.input,
        output: typeof data.usageMetadata.candidatesTokenCount === 'number' ? data.usageMetadata.candidatesTokenCount : usageOutputOf(text),
      };
    }
  }
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: finish ?? 'end_turn', stop_sequence: null },
    usage: { output_tokens: usage?.output ?? usageOutputOf(text) },
  });
  yield sseEvent('message_stop', { type: 'message_stop' });
  void started;
}
