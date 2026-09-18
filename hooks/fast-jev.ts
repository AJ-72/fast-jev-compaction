import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import {
  appendAuditRecord,
  buildAuditRecord,
  parseAuditLog,
  summarizeAuditLog,
  type AuditFs,
  type AuditOutcome,
  type AuditRecord,
} from '../src/audit.js';
import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
  auditLog: true,
  auditPath: '',
};

/** This plugin's version, recorded with every audit entry. */
const PLUGIN_VERSION = '0.3.1';

/**
 * Default audit log location. Kept beside Claude Code's own configuration so
 * it survives plugin upgrades and is findable without knowing the cache path.
 */
const DEFAULT_AUDIT_PATH = '~/.claude/fast-jev-audit.jsonl';

/**
 * Expands a leading `~`, or throws when the home directory is unknown.
 *
 * Stripping the `~` and writing a relative path instead would put the log in
 * whatever directory the session happens to run in, so records would scatter
 * across projects and `/jev-audit` would read a different file than the one
 * just written - the log would look empty while appearing to work. Failing
 * here is recoverable (the caller logs it and compaction continues); a log
 * silently written somewhere else is not.
 */
export function resolveAuditPath(path: string, home: string | undefined): string {
  if (!path.startsWith('~')) return path;
  if (!home) {
    throw new Error(
      'cannot expand ~ in the audit path: neither HOME nor USERPROFILE is set; set auditPath to an absolute path',
    );
  }
  return `${home.replace(/[/\\]$/, '')}/${path.slice(1).replace(/^[/\\]/, '')}`;
}

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  compactAtPercent: number;
  minReductionRatio: number;
  model: string;
  /** Write a durable record of every compaction attempt. Default true. */
  auditLog: boolean;
  /** Override the audit log location. Empty means the default path. */
  auditPath: string;
};

function optionBoolean(options: PluginOptions, key: string, fallback: boolean): boolean {
  const value = options[key];
  return typeof value === 'boolean' ? value : fallback;
}

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
    auditLog: optionBoolean(options, 'auditLog', HOOK_DEFAULTS.auditLog),
    auditPath: optionString(options, 'auditPath') ?? HOOK_DEFAULTS.auditPath,
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  return config;
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(fetchFn: HookFetch, apiKey: string, model: string): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/** Runs the library over a session transcript; throws when the key is missing or Jev fails. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
): Promise<SessionCompaction> {
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  const result = await compact(messages, jevAsker(fetchFn, config.apiKey, config.model), config);
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

/**
 * Picks the API key from the three places it may live, in precedence order.
 *
 * Takes the already-read env value and settings rather than the engine: `$`
 * may not be bound to a name or passed, so every engine call is made at the
 * hook's own call site and only plain values cross this boundary.
 */
export function pickApiKey(
  config: HookConfig,
  fromEnv: string | undefined,
  settings: Readonly<Record<string, unknown>>,
): string | undefined {
  if (config.apiKey) return config.apiKey;
  if (fromEnv) return fromEnv;
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

/** How long a fallback notice stays on screen. */
const NOTIFY_TIMEOUT_MS = 15_000;

/**
 * Resolves the audit log path from the configured value and the home dir.
 *
 * Takes `home` as a value rather than reading it from the engine: `$` may
 * not be bound or passed, so the hook reads HOME/USERPROFILE itself.
 */
export function auditLogPath(config: HookConfig, home: string | undefined): string {
  const configured = config.auditPath || DEFAULT_AUDIT_PATH;
  if (!configured.startsWith('~')) return configured;
  return resolveAuditPath(configured, home);
}

/**
 * Builds one audit record and appends it to the log.
 *
 * Takes the clock reading, session id, resolved path and an `AuditFs` as
 * plain values: `$` is the engine boundary and may not be bound to a name,
 * passed or read, so every engine call happens at the hook's own call site
 * and this function stays drivable by the tests without an engine.
 *
 * Deliberately swallows every failure and reports it in the returned
 * `error`: this is observability, and a plugin that broke compaction
 * because it could not write its own log would be worse than one with no
 * log at all.
 */
export async function recordAttempt(
  fs: AuditFs,
  config: HookConfig,
  input: {
    outcome: AuditOutcome;
    now: number;
    sessionId?: string;
    path: string;
    result?: CompactResult;
    fallbackReason?: string;
  },
): Promise<{ record?: AuditRecord; error?: string }> {
  if (!config.auditLog) return {};
  try {
    const record = buildAuditRecord({
      timestamp: new Date(input.now).toISOString(),
      pluginVersion: PLUGIN_VERSION,
      ...(input.sessionId ? { sessionId: input.sessionId } : {}),
      outcome: input.outcome,
      model: config.model,
      ...(input.result ? { result: input.result } : {}),
      ...(input.fallbackReason ? { fallbackReason: input.fallbackReason } : {}),
    });
    const written = await appendAuditRecord(fs, input.path, record);
    if (!written.written) return { record, error: written.error };
    return { record };
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

/** Renders the audit log for `/jev-audit`. */
export function formatAuditReport(records: readonly AuditRecord[], path: string): string {
  if (records.length === 0) {
    return `No compaction attempts recorded yet.\nLog: ${path}\n\nThe log is written when a compaction runs, so an empty log means none has happened in this install.`;
  }
  const summary = summarizeAuditLog(records);
  const lines: string[] = [
    `fast-jev-audit  (${path})`,
    '',
    `Attempts recorded:   ${summary.total}`,
    `Jev applied:         ${summary.jevApplied}`,
    `Below min reduction: ${summary.belowMinReduction}`,
    `Errors:              ${summary.errors}`,
  ];
  if (summary.meanAppliedRatio !== null) {
    lines.push(`Mean reduction:      ${Math.round(summary.meanAppliedRatio * 100)}% (applied only)`);
  }
  if (summary.fallbackReasons.length > 0) {
    lines.push('', 'Fallback reasons seen (most recent first):');
    for (const reason of summary.fallbackReasons.slice(0, 5)) lines.push(`  - ${reason}`);
  }
  lines.push('', 'Most recent attempts:');
  for (const record of records.slice(-10)) {
    const when = record.timestamp.slice(0, 19).replace('T', ' ');
    const ratio = record.reduction ? `${Math.round(record.reduction.ratio * 100)}%` : '-';
    const proof = record.jev
      ? `jev: ${record.jev.requests} req, ${record.jev.callsJudged} calls judged, ${record.jev.ms}ms`
      : 'jev: not reached';
    lines.push(`  ${when}  ${record.outcome.padEnd(20)} ${ratio.padStart(5)}  ${proof}`);
  }
  const withScores = records.filter((r) => r.sample && r.sample.length > 0).pop();
  if (withScores?.sample) {
    lines.push(
      '',
      `Jev scores from ${withScores.timestamp.slice(0, 19).replace('T', ' ')} (probabilities, not computable locally):`,
    );
    for (const s of withScores.sample.slice(0, 8)) {
      lines.push(`  ${s.id} ${s.tool.padEnd(12)} call=${s.keepCall.toFixed(2)} result=${s.keepResult.toFixed(2)} -> ${s.action}`);
    }
  }
  return lines.join('\n');
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;

  on('session.compact', async ($, event, next) => {
    // Every engine call is spelled `$.noun.method(...)` here at the call
    // site, and only plain values are handed to the helpers: `$` is the
    // engine boundary and may not be bound to a name, passed or read.
    const writeRecord = async (
      config: HookConfig,
      input: { outcome: AuditOutcome; result?: CompactResult; fallbackReason?: string },
    ) => {
      if (!config.auditLog) return;
      try {
        const [now, sessionId, home] = await Promise.all([
          $.clock.now(),
          $.session.id().catch(() => undefined),
          $.env.get('HOME').then((value) => value ?? $.env.get('USERPROFILE')),
        ]);
        const path = auditLogPath(config, home);
        const fs: AuditFs = {
          read: (target) => $.fs.read(target),
          write: (target, text) => $.fs.write(target, text),
          exists: (target) => $.fs.exists(target),
        };
        const { error } = await recordAttempt(fs, config, { ...input, now, sessionId, path });
        if (error) $.ui.log(`audit log not written (${error})`);
      } catch (error) {
        $.ui.log(`audit log skipped (${error instanceof Error ? error.message : String(error)})`);
      }
    };

    const notify = (text: string) => {
      $.ui.log(text);
      $.ui.toast(text, { timeoutMs: NOTIFY_TIMEOUT_MS });
    };

    try {
      const [fromEnv, settings] = await Promise.all([
        $.env.get('TYPESAFE_API_KEY'),
        $.settings.read(),
      ]);
      const config = { ...configured, apiKey: pickApiKey(configured, fromEnv, settings) };
      const { result, messages } = await compactSession(event.messages, config, async (url, init) => {
        const response = await $.http.fetch(url, init);
        return { status: response.status, ok: response.ok, text: response.text };
      });
      for (const line of decisionLogLines(result)) $.ui.log(line);
      if (reductionRatio(result) < config.minReductionRatio) {
        const reason = `below ${percent(config.minReductionRatio)} minimum: ${summarize(result)}`;
        await writeRecord(config, { outcome: 'below_min_reduction', result, fallbackReason: reason });
        notify(`fallback to built-in summary (${reason})`);
        return next(event);
      }
      await writeRecord(config, { outcome: 'jev_applied', result });
      notify(
        `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`,
      );
      return { messages };
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      // `configured`, not `config`: the failure may have been building `config`
      // itself (a rejected key lookup), and the audit settings are the same.
      await writeRecord(configured, { outcome: 'error', fallbackReason: reason });
      notify(`fallback to built-in summary (${reason})`);
      return next(event);
    }
  });

  // `/jev-audit` reads the log back. Registering on session.start rather than
  // at module load keeps it out of the way when the plugin is disabled mid-run.
  on('session.start', async ($, event, next) => {
    try {
      await $.command.register({
        name: 'jev-audit',
        description: 'Show proof of whether Jev handled recent compactions.',
      });
    } catch (error) {
      $.ui.log(`/jev-audit unavailable (${error instanceof Error ? error.message : String(error)})`);
    }
    return next(event);
  });

  on('command.run', async ($, event, next) => {
    if (event.command !== 'jev-audit') return next(event);
    try {
      const home = (await $.env.get('HOME')) ?? (await $.env.get('USERPROFILE'));
      const path = auditLogPath(configured, home);
      if (!(await $.fs.exists(path))) {
        return { text: formatAuditReport([], path) };
      }
      const records = parseAuditLog(await $.fs.read(path));
      return { text: formatAuditReport(records, path) };
    } catch (error) {
      return {
        text: `Could not read the audit log (${error instanceof Error ? error.message : String(error)}).`,
      };
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (compacting) return next(event);
    try {
      const { context } = await $.session.usage();
      if ((context.percent ?? 0) < configured.compactAtPercent) return next(event);
      compacting = true;
      await $.session.compact();
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
