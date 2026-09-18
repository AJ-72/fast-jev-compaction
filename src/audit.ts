/**
 * Durable evidence that this plugin ran.
 *
 * The hook already reports what it did through `$.ui.log` and `$.ui.toast`,
 * but both are transient: once the toast fades and the session ends, nothing
 * on disk says whether Jev compacted the transcript, whether it fell back to
 * the built-in summarizer, or why. That matters most in exactly the case you
 * would want to know about — a missing key or an unreachable API silently
 * degrades every compaction to the built-in path, and the only sign is a
 * toast nobody was watching.
 *
 * This module records one JSON line per compaction attempt, whatever the
 * outcome, so the question "did Jev actually run?" is answerable after the
 * fact instead of inferred from timing.
 */

import type { CompactResult } from './types.js';

/** Where an attempt ended up. */
export type AuditOutcome =
  /** Jev answered and its pruned transcript replaced the history. */
  | 'jev_applied'
  /** Jev answered but reduced too little; the built-in summarizer ran instead. */
  | 'below_min_reduction'
  /** Something failed (no key, API error, bad response); built-in summarizer ran. */
  | 'error';

/** One compaction attempt, as written to the audit log. */
export interface AuditRecord {
  /** ISO 8601, from the engine's clock. */
  timestamp: string;
  /** Schema version, so a reader can tell old records from new. */
  v: 1;
  plugin: string;
  pluginVersion: string;
  sessionId?: string;
  outcome: AuditOutcome;
  /** True only when Jev's output was actually used. */
  jevApplied: boolean;
  /** Present when a request reached Jev. */
  jev?: {
    model: string;
    /** How many HTTP requests the state needed. */
    requests: number;
    /** Wall-clock ms inside the library, including Jev round-trips. */
    ms: number;
    stateTokens: number;
    stateStage: string;
    /** Tool calls Jev was asked about (excludes pinned). */
    callsJudged: number;
  };
  /** Present whenever the library produced a result. */
  reduction?: {
    charsBefore: number;
    charsAfter: number;
    ratio: number;
    messagesBefore: number;
    messagesAfter: number;
    kept: number;
    resultsDropped: number;
    callsDropped: number;
    pinned: number;
  };
  /** Why the built-in summarizer ran; absent on `jev_applied`. */
  fallbackReason?: string;
  /**
   * Per-call scores, the strongest evidence a real model answered: these are
   * probabilities from Jev, not anything the plugin could compute locally.
   * Capped so a long session cannot bloat the log.
   */
  sample?: {
    id: string;
    tool: string;
    action: string;
    keepCall: number;
    keepResult: number;
  }[];
}

/** How many per-call scores to record. Enough to be evidence, not a transcript. */
const SAMPLE_LIMIT = 20;

/** Keep the log bounded; oldest records are dropped past this. */
export const MAX_AUDIT_RECORDS = 500;

export function reductionRatioOf(result: CompactResult): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore > 0 ? 1 - charsAfter / charsBefore : 0;
}

/**
 * Builds the record for an attempt. Pure, so the hook stays thin and the
 * shape can be asserted directly in tests.
 */
export function buildAuditRecord(input: {
  timestamp: string;
  pluginVersion: string;
  sessionId?: string;
  outcome: AuditOutcome;
  model: string;
  result?: CompactResult;
  fallbackReason?: string;
}): AuditRecord {
  const { result } = input;
  const record: AuditRecord = {
    timestamp: input.timestamp,
    v: 1,
    plugin: 'fast-jev-audit',
    pluginVersion: input.pluginVersion,
    outcome: input.outcome,
    jevApplied: input.outcome === 'jev_applied',
  };
  if (input.sessionId) record.sessionId = input.sessionId;
  if (input.fallbackReason) record.fallbackReason = input.fallbackReason;

  if (result) {
    const { stats } = result;
    // `requests > 0` is the discriminator for "Jev was actually contacted":
    // an error before the first request leaves a result with none.
    if (stats.requests > 0) {
      record.jev = {
        model: input.model,
        requests: stats.requests,
        ms: stats.ms,
        stateTokens: stats.stateTokens,
        stateStage: stats.stateStage,
        callsJudged: stats.calls - stats.pinned,
      };
    }
    record.reduction = {
      charsBefore: stats.charsBefore,
      charsAfter: stats.charsAfter,
      ratio: Number(reductionRatioOf(result).toFixed(4)),
      messagesBefore: stats.messagesBefore,
      messagesAfter: stats.messagesAfter,
      kept: stats.kept,
      resultsDropped: stats.resultsDropped,
      callsDropped: stats.callsDropped,
      pinned: stats.pinned,
    };
    const scored = result.decisions.filter((d) => d.reason !== 'pinned');
    if (scored.length > 0) {
      record.sample = scored.slice(0, SAMPLE_LIMIT).map((d) => ({
        id: d.id,
        tool: d.tool,
        action: d.action,
        keepCall: Number(d.keepCall.toFixed(4)),
        keepResult: Number(d.keepResult.toFixed(4)),
      }));
    }
  }
  return record;
}

/** The minimum of `$.fs` this module needs, so tests need no engine. */
export interface AuditFs {
  read: (path: string) => Promise<string>;
  write: (path: string, text: string) => Promise<void>;
  exists: (path: string) => Promise<boolean>;
}

/**
 * Appends a record to the JSONL audit log.
 *
 * The engine's `fs` has no append, so this reads, trims to the newest
 * `MAX_AUDIT_RECORDS`, and rewrites. Compaction is infrequent and the file is
 * bounded, so the cost is irrelevant next to a Jev round-trip.
 *
 * Never throws: an audit failure must not turn a working compaction into a
 * fallback. The caller reports the problem through the UI instead.
 */
export async function appendAuditRecord(
  fs: AuditFs,
  path: string,
  record: AuditRecord,
): Promise<{ written: boolean; error?: string }> {
  try {
    let lines: string[] = [];
    if (await fs.exists(path)) {
      const existing = await fs.read(path);
      lines = existing.split('\n').filter((line) => line.trim().length > 0);
    }
    lines.push(JSON.stringify(record));
    if (lines.length > MAX_AUDIT_RECORDS) lines = lines.slice(-MAX_AUDIT_RECORDS);
    await fs.write(path, `${lines.join('\n')}\n`);
    return { written: true };
  } catch (error) {
    return { written: false, error: error instanceof Error ? error.message : String(error) };
  }
}

/** Parses an audit log, skipping malformed lines rather than failing. */
export function parseAuditLog(text: string): AuditRecord[] {
  const records: AuditRecord[] = [];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    try {
      const parsed = JSON.parse(line) as AuditRecord;
      if (parsed && typeof parsed === 'object' && 'outcome' in parsed) records.push(parsed);
    } catch {
      // A truncated final line (killed mid-write) is expected; skip it.
    }
  }
  return records;
}

export interface AuditSummary {
  total: number;
  jevApplied: number;
  belowMinReduction: number;
  errors: number;
  /** Mean reduction ratio over applied compactions only. */
  meanAppliedRatio: number | null;
  lastOutcome?: AuditOutcome;
  lastTimestamp?: string;
  /** Distinct fallback reasons seen, most recent first. */
  fallbackReasons: string[];
}

/** Aggregates a log into the answer to "is Jev actually doing the work?". */
export function summarizeAuditLog(records: readonly AuditRecord[]): AuditSummary {
  const applied = records.filter((r) => r.outcome === 'jev_applied');
  const ratios = applied
    .map((r) => r.reduction?.ratio)
    .filter((r): r is number => typeof r === 'number');
  const reasons: string[] = [];
  for (let i = records.length - 1; i >= 0; i -= 1) {
    const reason = records[i]?.fallbackReason;
    if (reason && !reasons.includes(reason)) reasons.push(reason);
  }
  const last = records[records.length - 1];
  const summary: AuditSummary = {
    total: records.length,
    jevApplied: applied.length,
    belowMinReduction: records.filter((r) => r.outcome === 'below_min_reduction').length,
    errors: records.filter((r) => r.outcome === 'error').length,
    meanAppliedRatio:
      ratios.length > 0 ? Number((ratios.reduce((a, b) => a + b, 0) / ratios.length).toFixed(4)) : null,
    fallbackReasons: reasons,
  };
  if (last) {
    summary.lastOutcome = last.outcome;
    summary.lastTimestamp = last.timestamp;
  }
  return summary;
}
