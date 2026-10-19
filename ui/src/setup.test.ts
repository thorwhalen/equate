import { describe, expect, it } from 'vitest';
import { parseSetup } from './setup';

describe('parseSetup', () => {
  it('reads the drafts', () => {
    expect(parseSetup({ k: '5', floor: '' })).toEqual({ ok: true, value: { k: 5, floor: null } });
    expect(parseSetup({ k: ' 12 ', floor: ' 0.5 ' })).toEqual({ ok: true, value: { k: 12, floor: 0.5 } });
    expect(parseSetup({ k: '3', floor: '-1.5' })).toEqual({ ok: true, value: { k: 3, floor: -1.5 } });
  });

  it('names each bad field instead of rewriting it', () => {
    const r = parseSetup({ k: '', floor: 'high' });
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.errors.k).toBe('Candidates per item is required');
      expect(r.errors.floor).toMatch(/decimal number/);
    }
    expect(parseSetup({ k: '2.5', floor: '' }).ok).toBe(false);
    expect(parseSetup({ k: '99', floor: '' }).ok).toBe(false);
  });

  it('reads numbers the way Python float() does', () => {
    for (const bad of ['0x10', '0b1', '0o7', 'Infinity', '1e400', '0,5', '1_000']) {
      expect(parseSetup({ k: '5', floor: bad }).ok, bad).toBe(false);
    }
    for (const [ok, v] of [['.5', 0.5], ['5.', 5], ['1e-3', 0.001], ['+2', 2]] as const) {
      expect(parseSetup({ k: '5', floor: ok }), ok).toEqual({ ok: true, value: { k: 5, floor: v } });
    }
    expect(parseSetup({ k: '0x5', floor: '' }).ok).toBe(false);
  });
});
