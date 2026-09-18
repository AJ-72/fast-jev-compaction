import { describe, expect, it } from 'vitest';
import {
  appendAuditRecord,
  buildAuditRecord,
  MAX_AUDIT_RECORDS,
  parseAuditLog,
  summarizeAuditLog,
  type AuditFs,
  type AuditRecord,
} from '../src/audit.js';
import {
  auditLogPath,
  formatAuditReport,
  recordAttempt,
  resolveAuditPath,
} from '../hooks/fast-jev.ts';
import type { CompactResult } from '../src/types.js';

/** An in-memory `$.fs`, enough for the audit module. */
function memoryFs(seed: Record<string, string> = {}) {
  const files = new Map<string, string>(Object.entries(seed));
  const fs: AuditFs & { files: Map<string, string> } = {
    files,
    read: async (path) => {
      const text = files.get(path);
      if (text === undefined) throw new Error(`ENOENT: ${path}`);
      return text;
    },
    write: async (path, text) => {
      files.set(path, text);
    },
    exists: async (path) => files.has(path),
  };
  return fs;
}

function result(overrides: Partial<CompactResult['stats']> = {}): CompactResult {
  return {
    messages: [],
    decisions: [
      { id: 't1', tool: 'Read', action: 'drop_result', reason: 'result_dropped', keepCall: 0.81, keepResult: 0.12 },
      { id: 't2', tool: 'Bash', action: 'keep', reason: 'kept', keepCall: 0.97, keepResult: 0.88 },
      { id: 't3', tool: 'Grep', action: 'keep', reason: 'pinned', keepCall: 1, keepResult: 1 },
    ],
    stats: {
      messagesBefore: 20,
      messagesAfter: 18,
      charsBefore: 10_000,
      charsAfter: 4_000,
      calls: 3,
      kept: 1,
      resultsDropped: 1,
      callsDropped: 0,
      pinned: 1,
      stateTokens: 8_000,
      stateStage: 'full',
      requests: 2,
      ms: 1_340,
      ...overrides,
    },
  };
}

describe('buildAuditRecord', () => {
  it('marks an applied compaction and records Jev-only evidence', () => {
    const record = buildAuditRecord({
      timestamp: '2026-09-18T12:00:00.000Z',
      pluginVersion: '0.3.0',
      sessionId: 'abc',
      outcome: 'jev_applied',
      model: 'jev-latest',
      result: result(),
    });

    expect(record.jevApplied).toBe(true);
    expect(record.jev).toEqual({
      model: 'jev-latest',
      requests: 2,
      ms: 1_340,
      stateTokens: 8_000,
      stateStage: 'full',
      callsJudged: 2,
    });
    expect(record.reduction?.ratio).toBe(0.6);
    // Pinned decisions carry no Jev judgement, so they are not evidence.
    expect(record.sample).toHaveLength(2);
    expect(record.sample?.[0]).toMatchObject({ id: 't1', keepCall: 0.81, keepResult: 0.12 });
  });

  it('does not claim Jev was reached when no request was made', () => {
    const record = buildAuditRecord({
      timestamp: '2026-09-18T12:00:00.000Z',
      pluginVersion: '0.3.0',
      outcome: 'error',
      model: 'jev-latest',
      result: result({ requests: 0 }),
    });
    expect(record.jev).toBeUndefined();
    expect(record.jevApplied).toBe(false);
  });

  it('records a fallback reason with no result at all', () => {
    const record = buildAuditRecord({
      timestamp: '2026-09-18T12:00:00.000Z',
      pluginVersion: '0.3.0',
      outcome: 'error',
      model: 'jev-latest',
      fallbackReason: 'TYPESAFE_API_KEY is not configured',
    });
    expect(record.jevApplied).toBe(false);
    expect(record.reduction).toBeUndefined();
    expect(record.fallbackReason).toBe('TYPESAFE_API_KEY is not configured');
  });
});

describe('appendAuditRecord', () => {
  it('appends across calls and keeps the file valid JSONL', async () => {
    const fs = memoryFs();
    const base = { timestamp: '2026-09-18T12:00:00.000Z', v: 1 as const, plugin: 'fast-jev-audit', pluginVersion: '0.3.0', jevApplied: true };

    await appendAuditRecord(fs, 'log.jsonl', { ...base, outcome: 'jev_applied' });
    await appendAuditRecord(fs, 'log.jsonl', { ...base, outcome: 'error', jevApplied: false });

    const parsed = parseAuditLog(fs.files.get('log.jsonl') ?? '');
    expect(parsed.map((r) => r.outcome)).toEqual(['jev_applied', 'error']);
  });

  it('caps the log so it cannot grow without bound', async () => {
    const fs = memoryFs();
    const record: AuditRecord = {
      timestamp: '2026-09-18T12:00:00.000Z',
      v: 1,
      plugin: 'fast-jev-audit',
      pluginVersion: '0.3.0',
      outcome: 'jev_applied',
      jevApplied: true,
    };
    for (let i = 0; i < MAX_AUDIT_RECORDS + 10; i += 1) {
      await appendAuditRecord(fs, 'log.jsonl', { ...record, sessionId: `s${i}` });
    }
    const parsed = parseAuditLog(fs.files.get('log.jsonl') ?? '');
    expect(parsed).toHaveLength(MAX_AUDIT_RECORDS);
    // The newest survive, not the oldest.
    expect(parsed[parsed.length - 1]?.sessionId).toBe(`s${MAX_AUDIT_RECORDS + 9}`);
  });

  it('reports a write failure instead of throwing', async () => {
    const fs: AuditFs = {
      read: async () => '',
      exists: async () => false,
      write: async () => {
        throw new Error('disk full');
      },
    };
    const outcome = await appendAuditRecord(fs, 'log.jsonl', {
      timestamp: 't',
      v: 1,
      plugin: 'fast-jev-audit',
      pluginVersion: '0.3.0',
      outcome: 'jev_applied',
      jevApplied: true,
    });
    expect(outcome).toEqual({ written: false, error: 'disk full' });
  });

  it('skips a truncated final line rather than failing', () => {
    const good = JSON.stringify({ outcome: 'jev_applied', timestamp: 't' });
    expect(parseAuditLog(`${good}\n{"outcome":"err`)).toHaveLength(1);
  });
});

describe('summarizeAuditLog', () => {
  it('counts outcomes and surfaces distinct fallback reasons newest first', () => {
    const records = parseAuditLog(
      [
        JSON.stringify({ outcome: 'jev_applied', timestamp: '1', reduction: { ratio: 0.6 } }),
        JSON.stringify({ outcome: 'below_min_reduction', timestamp: '2', fallbackReason: 'too small' }),
        JSON.stringify({ outcome: 'error', timestamp: '3', fallbackReason: 'no key' }),
        JSON.stringify({ outcome: 'jev_applied', timestamp: '4', reduction: { ratio: 0.8 } }),
      ].join('\n'),
    );
    const summary = summarizeAuditLog(records);
    expect(summary).toMatchObject({
      total: 4,
      jevApplied: 2,
      belowMinReduction: 1,
      errors: 1,
      meanAppliedRatio: 0.7,
      lastOutcome: 'jev_applied',
    });
    expect(summary.fallbackReasons).toEqual(['no key', 'too small']);
  });

  it('reports no mean when nothing was applied', () => {
    expect(summarizeAuditLog([]).meanAppliedRatio).toBeNull();
  });
});

describe('resolveAuditPath', () => {
  it('expands ~ against HOME or USERPROFILE', () => {
    expect(resolveAuditPath('~/.claude/a.jsonl', 'C:/Users/x')).toBe('C:/Users/x/.claude/a.jsonl');
    expect(resolveAuditPath('~/.claude/a.jsonl', '/home/y/')).toBe('/home/y/.claude/a.jsonl');
  });

  it('leaves an absolute path alone', () => {
    expect(resolveAuditPath('/var/log/a.jsonl', '/home/y')).toBe('/var/log/a.jsonl');
  });

  it('refuses to guess when the home directory is unknown', () => {
    // Writing a relative path instead would scatter logs per working
    // directory and make /jev-audit read a different file than was written.
    expect(() => resolveAuditPath('~/.claude/a.jsonl', undefined)).toThrow(/HOME|USERPROFILE/);
  });
});

const NOW = Date.parse('2026-09-18T12:00:00.000Z');

const config = {
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  model: 'jev-latest',
  auditLog: true,
  auditPath: '',
};

describe('recordAttempt', () => {
  it('writes an applied record to the default path', async () => {
    const fs = memoryFs();
    const { record } = await recordAttempt(fs, config, {
      outcome: 'jev_applied',
      result: result(),
      now: NOW,
      sessionId: 'session-1',
      path: auditLogPath(config, '/home/t'),
    });

    expect(record?.jevApplied).toBe(true);
    expect(record?.sessionId).toBe('session-1');
    const written = fs.files.get('/home/t/.claude/fast-jev-audit.jsonl');
    expect(written).toBeDefined();
    expect(parseAuditLog(written ?? '')[0]?.outcome).toBe('jev_applied');
  });

  it('writes nothing when auditing is turned off', async () => {
    const fs = memoryFs();
    const { record } = await recordAttempt(fs, { ...config, auditLog: false }, {
      outcome: 'jev_applied',
      result: result(),
      now: NOW,
      path: auditLogPath(config, '/home/t'),
    });
    expect(record).toBeUndefined();
    expect(fs.files.size).toBe(0);
  });

  it('never throws when the filesystem fails, and reports the reason', async () => {
    const fs: AuditFs = {
      read: async () => '',
      exists: async () => false,
      write: async () => {
        throw new Error('read-only volume');
      },
    };
    // The caller logs this through `$.ui.log`; the contract here is that the
    // reason comes back rather than escaping as a throw.
    const { error } = await recordAttempt(fs, config, {
      outcome: 'jev_applied',
      result: result(),
      now: NOW,
      path: auditLogPath(config, '/home/t'),
    });
    expect(error).toContain('read-only volume');
  });

  it('honours an explicit auditPath', async () => {
    const fs = memoryFs();
    const withPath = { ...config, auditPath: '/tmp/custom.jsonl' };
    await recordAttempt(fs, withPath, {
      outcome: 'error',
      fallbackReason: 'no key',
      now: NOW,
      path: auditLogPath(withPath, '/home/t'),
    });
    expect(fs.files.has('/tmp/custom.jsonl')).toBe(true);
  });
});

describe('auditLogPath', () => {
  it('throws when the home directory is unknown', () => {
    // Moved here from recordAttempt: resolving the path is now the hook's
    // job, so this is where an unknown home has to surface. The hook catches
    // it and reports through `$.ui.log`.
    expect(() => auditLogPath(config, undefined)).toThrow(/HOME|USERPROFILE/);
  });

  it('leaves an absolute path alone, home or not', () => {
    expect(auditLogPath({ ...config, auditPath: '/tmp/a.jsonl' }, undefined)).toBe('/tmp/a.jsonl');
  });
});