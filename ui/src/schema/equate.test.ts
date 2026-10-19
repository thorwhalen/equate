import { describe, expect, it } from 'vitest';
import fields from './fields.json';
import { BOUNDARY, SessionView } from './equate';

describe('Zod schemas mirror the Python boundary types', () => {
  it('covers exactly the Python types', () => {
    expect(Object.keys(BOUNDARY).sort()).toEqual(Object.keys(fields).sort());
  });

  for (const [name, pyFields] of Object.entries(fields)) {
    it(`${name} has the same fields as the dataclass`, () => {
      const schema = BOUNDARY[name as keyof typeof BOUNDARY];
      expect(Object.keys(schema.shape).sort()).toEqual([...(pyFields as string[])].sort());
    });
  }

  it('parses a service payload', () => {
    const payload = {
      session_id: 's1', n_left: 1, n_right: 1, sense: 'maximize', k: 5,
      assignment: [{ item: 0, candidate: 0, left: 'a', right: 'a', score: 1, status: 'proposed' }],
      review: [{ item: 0, left: 'a', assigned: 0, status: 'proposed', margin: null, candidates: [] }],
      edits: [], labels: [], accounting: { tiers: { score: { calls: 1, seconds: 0.01, items: 1 } } }, floor: null,
    };
    expect(SessionView.parse(payload).review[0].margin).toBeNull();
  });
});
