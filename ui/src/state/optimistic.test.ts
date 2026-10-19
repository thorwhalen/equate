import { describe, expect, it } from 'vitest';
import type { SessionView } from '../schema/equate';
import { applyOptimistic, reconcile } from './optimistic';

const cand = (candidate: number, score: number) => ({
  candidate,
  right: `r${candidate}`,
  score,
  explanation: null,
});

const session: SessionView = {
  session_id: 's',
  n_left: 2,
  n_right: 2,
  sense: 'maximize',
  k: 2,
  assignment: [
    { item: 0, candidate: 0, left: 'l0', right: 'r0', score: 0.9, status: 'proposed' },
    { item: 1, candidate: 1, left: 'l1', right: 'r1', score: 0.8, status: 'proposed' },
  ],
  review: [
    { item: 1, left: 'l1', assigned: 1, status: 'proposed', margin: 0.1, candidates: [cand(1, 0.8), cand(0, 0.7)] },
    { item: 0, left: 'l0', assigned: 0, status: 'proposed', margin: 0.5, candidates: [cand(0, 0.9), cand(1, 0.4)] },
  ],
  edits: [],
  labels: [],
  accounting: { tiers: {} },
  floor: null,
};

describe('applyOptimistic', () => {
  it('confirm takes the candidate and displaces its holder', () => {
    const s = applyOptimistic(session, { action: 'confirm', item: 1, candidate: 0, source: 'human' });
    expect(s.assignment).toEqual([
      { item: 1, candidate: 0, left: 'l1', right: 'r0', score: 0.7, status: 'confirmed' },
    ]);
    expect(s.review.map((r) => [r.item, r.assigned, r.status])).toEqual([
      [1, 0, 'confirmed'],
      [0, null, 'unmatched'],
    ]);
    expect(s.edits).toHaveLength(1);
    expect(session.edits).toHaveLength(0); // immutable
  });

  it('reject unassigns the rejected pair only', () => {
    const s = applyOptimistic(session, { action: 'reject', item: 0, candidate: 0, source: 'human' });
    expect(s.assignment.map((p) => p.item)).toEqual([1]);
    expect(s.review.find((r) => r.item === 0)?.status).toBe('unmatched');
  });
});

describe('reconcile', () => {
  it('takes the server assignment and keeps queue order', () => {
    const optimistic = applyOptimistic(session, { action: 'confirm', item: 1, candidate: 0, source: 'human' });
    const s = reconcile(optimistic, {
      session_id: 's',
      assignment: [
        { item: 0, candidate: 1, left: 'l0', right: 'r1', score: 0.4, status: 'proposed' },
        { item: 1, candidate: 0, left: 'l1', right: 'r0', score: 0.7, status: 'confirmed' },
      ],
      changed: [0, 1],
      accounting: { tiers: { reoptimize: { calls: 1, seconds: 0.001, items: 4 } } },
    });
    expect(s.review.map((r) => [r.item, r.assigned, r.status])).toEqual([
      [1, 0, 'confirmed'],
      [0, 1, 'proposed'],
    ]);
    expect(s.accounting.tiers.reoptimize.calls).toBe(1);
  });
});
