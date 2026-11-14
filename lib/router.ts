// router.ts — which free model serves a request, and what to try next when one refuses.
//
// The trial never publishes a provider's per-trial budget, its failure schedule, or its outage:
// the entry learns those only from the refusals it sees. So the router keeps, per provider, what
// it has observed — how many calls it made, how many refusals in a row — and derives every choice
// from the brief, the requests, and those observations. Nothing here reads the wall clock or an
// unseeded random source, so a replay makes the same choices.
//
// Ranking, for one request:
//   1. Only a free model that meets the request's requirements may serve it. A paid model is
//      never a legal pick, whatever its rating.
//   2. A JSON-mode request on the OpenAI wire carries response_format, which the Anthropic wire
//      cannot carry (an Anthropic provider echoes json_mode absent, and the echo check fails).
//      Such a request may only go to an OpenAI-wire or Gemini-wire provider. (An Anthropic
//      client never asks for JSON mode on its own wire, so its requests may use any wire.)
//   3. By rating, best first — quality is 60% of the score. The weight-2 requests arrive first
//      in the manifest, so greedily taking the best available also serves the heavy requests
//      best, with no look-ahead needed.
//   4. Ties in rating go to the provider this trial has called least so far, then to the
//      provider id, then the model id: the same model id on two providers (glm-4.7-flash on
//      Z.ai and Cloudflare, qwen-3.8-27b on Groq and Cerebras) spreads its load instead of
//      draining one provider's budget.
//   5. A provider that refused twice in a row is cooling down: it is tried only when no other
//      candidate is left, until three requests have been served since its last refusal. A
//      budget spent or an outage refuses every later call, so two in a row is the signal; a
//      lone scheduled failure is transient, and the provider is tried again at once.
//
// A refusal costs a call (the tie-break counts provider calls), so within one request each
// (provider, model) is tried at most once, and the router never re-tries the same provider for
// the same request: the next candidate is a different provider's model, which is also what the
// failover column wants — a request whose first call failed, still served.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
  type Wire,
} from './providers.ts';

/** One (provider, model) that may serve a request, with its sealed rating. */
export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** What the trial has taught the router about one provider so far. */
interface ProviderHealth {
  /** Inference calls made to this provider (catalog probes do not count). */
  calls: number;
  /** Refusals in a row from this provider; a success resets it to zero. */
  refusals: number;
  /** Manifest index of the request whose call last refused here, or -1. */
  lastRefusalReq: number;
}

/** How many requests must be served after a provider's last refusal before it is tried again. */
const COOLDOWN_REQUESTS = 3;
/** Refusals in a row after which a provider cools down. */
const COOLDOWN_AFTER = 2;

export class Router {
  private readonly probes: readonly Probe[];
  private readonly ratings: RatingsFile | null;
  private readonly health = new Map<string, ProviderHealth>();

  constructor(probes: readonly Probe[], ratings: RatingsFile | null) {
    this.probes = probes;
    this.ratings = ratings;
    for (const probe of probes) {
      this.health.set(probe.provider.id, { calls: 0, refusals: 0, lastRefusalReq: -1 });
    }
  }

  private healthOf(providerId: string): ProviderHealth {
    let h = this.health.get(providerId);
    if (h === undefined) {
      h = { calls: 0, refusals: 0, lastRefusalReq: -1 };
      this.health.set(providerId, h);
    }
    return h;
  }

  /** Whether a provider is cooling down at this point of the workload. */
  private cooling(providerId: string, requestIndex: number): boolean {
    const h = this.healthOf(providerId);
    return h.refusals >= COOLDOWN_AFTER && requestIndex - h.lastRefusalReq < COOLDOWN_REQUESTS;
  }

  /** Record a provider call's outcome. A success resets the refusal streak. */
  noteCall(providerId: string, requestIndex: number, ok: boolean): void {
    const h = this.healthOf(providerId);
    h.calls += 1;
    if (ok) {
      h.refusals = 0;
    } else {
      h.refusals += 1;
      h.lastRefusalReq = requestIndex;
    }
  }

  /** How many inference calls this trial has made to one provider. */
  callsTo(providerId: string): number {
    return this.healthOf(providerId).calls;
  }

  /**
   * The candidates for one request, best first.
   *
   * `clientApi` is the wire the client used; `jsonMode` is true when the client asked for JSON
   * mode on the OpenAI wire. Cooling-down providers sit at the back of the order, still tried
   * when nothing else is left.
   */
  candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean, requestIndex: number): Candidate[] {
    const ready: Candidate[] = [];
    const cooling: Candidate[] = [];
    for (const probe of this.probes) {
      if (!probe.ok) continue;
      // The Anthropic wire cannot carry an OpenAI JSON-mode request: its providers echo
      // json_mode absent, which fails the client's echo check.
      if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
      for (const model of probe.models) {
        const row = ratingOfModel(this.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
        const candidate: Candidate = { probe, model, rating: row?.rating_milli ?? 0 };
        if (this.cooling(probe.provider.id, requestIndex)) cooling.push(candidate);
        else ready.push(candidate);
      }
    }
    const byRank = (a: Candidate, b: Candidate): number => {
      if (b.rating !== a.rating) return b.rating - a.rating;
      const callsA = this.healthOf(a.probe.provider.id).calls;
      const callsB = this.healthOf(b.probe.provider.id).calls;
      if (callsA !== callsB) return callsA - callsB;
      if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    };
    ready.sort(byRank);
    cooling.sort(byRank);
    return [...ready, ...cooling];
  }

  /** The wire a provider speaks: the caller needs it to build the upstream request. */
  static wireOf(candidate: Candidate): Wire {
    return candidate.probe.provider.wire;
  }
}
