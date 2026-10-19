import { describe, expect, it } from 'vitest';
import { parseSetup } from './setup';

describe('parseSetup', () => {
  it('reads the drafts', () => {
    expect(parseSetup({ k: '5', threshold: '' })).toEqual({ ok: true, value: { k: 5, threshold: null } });
    expect(parseSetup({ k: ' 12 ', threshold: ' 0.5 ' })).toEqual({ ok: true, value: { k: 12, threshold: 0.5 } });
    expect(parseSetup({ k: '3', threshold: '-1.5' })).toEqual({ ok: true, value: { k: 3, threshold: -1.5 } });
  });

  it('names each bad field instead of rewriting it', () => {
    const r = parseSetup({ k: '', threshold: 'high' });
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.errors.k).toMatch(/at least 1|number/);
      expect(r.errors.threshold).toMatch(/number/);
    }
    expect(parseSetup({ k: '2.5', threshold: '' }).ok).toBe(false);
    expect(parseSetup({ k: '99', threshold: '' }).ok).toBe(false);
  });
});
