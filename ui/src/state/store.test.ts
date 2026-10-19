import { describe, expect, it, vi } from 'vitest';
import type { Client } from '../api/client';
import type { SessionView } from '../schema/equate';
import { createReviewStore } from './store';

const session: SessionView = {
  session_id: 's1',
  n_left: 2,
  n_right: 2,
  sense: 'maximize',
  k: 2,
  assignment: [
    { item: 0, candidate: 0, left: 'a', right: 'a', score: 1, status: 'proposed' },
    { item: 1, candidate: 1, left: 'b', right: 'b', score: 1, status: 'proposed' },
  ],
  review: [
    { item: 0, left: 'a', assigned: 0, status: 'proposed', margin: 0.2, candidates: [] },
    { item: 1, left: 'b', assigned: 1, status: 'proposed', margin: 0.3, candidates: [] },
  ],
  edits: [],
  labels: [],
  accounting: { tiers: {} },
  floor: null,
};

function fakeClient(overrides: Partial<Client> = {}): Client {
  return {
    openSession: vi.fn(async () => session),
    getSession: vi.fn(async () => session),
    candidates: vi.fn(async () => []),
    edit: vi.fn(async () => ({
      session_id: 's1',
      assignment: [
        { item: 0, candidate: 1, left: 'a', right: 'b', score: 0.1, status: 'confirmed' as const },
        { item: 1, candidate: 0, left: 'b', right: 'a', score: 0.1, status: 'proposed' as const },
      ],
      changed: [0, 1],
      accounting: { tiers: {} },
    })),
    label: vi.fn(),
    next: vi.fn(async () => null),
    ...overrides,
  } as Client;
}

describe('review store', () => {
  it('opens a session and selects the most uncertain item', async () => {
    const store = createReviewStore(fakeClient());
    expect(await store.getState().open({ A: ['a', 'b'], B: ['a', 'b'] })).toBe('s1');
    expect(store.getState().status).toBe('ready');
    expect(store.getState().selected).toBe(0);
  });

  it('shows an edit at once, then the server re-optimization', async () => {
    const store = createReviewStore(fakeClient());
    await store.getState().open({ A: [], B: [] });
    const done = store.getState().edit({ action: 'confirm', item: 0, candidate: 1, source: 'human' });
    // optimistic: before the server answers
    expect(store.getState().session?.review[0].status).toBe('confirmed');
    expect(store.getState().pending).toEqual([0]);
    await done;
    expect(store.getState().pending).toEqual([]);
    expect(store.getState().flash).toEqual([0, 1]);
    expect(store.getState().session?.review[1].assigned).toBe(0);
  });

  it('re-reads the session when an edit fails', async () => {
    const client = fakeClient({ edit: vi.fn(async () => Promise.reject(new Error('nope'))) });
    const store = createReviewStore(client);
    await store.getState().open({ A: [], B: [] });
    await store.getState().edit({ action: 'reject', item: 0, candidate: 0, source: 'human' });
    expect(store.getState().error).toMatch(/not applied: nope/);
    expect(client.getSession).toHaveBeenCalledWith('s1');
    expect(store.getState().session?.review[0].status).toBe('proposed');
  });
});

describe('queued edits and labels', () => {
  it('keeps a queued edit on screen while an earlier one lands', async () => {
    const answers: Array<(v: unknown) => void> = [];
    const client = fakeClient({
      edit: vi.fn(() => new Promise((resolve) => answers.push(resolve))) as Client['edit'],
    });
    const store = createReviewStore(client);
    await store.getState().open({ A: [], B: [] });
    const first = store.getState().edit({ action: 'confirm', item: 0, candidate: 0, source: 'human' });
    store.getState().edit({ action: 'confirm', item: 1, candidate: 1, source: 'human' });
    await vi.waitFor(() => expect(answers).toHaveLength(1));
    // the server has answered only the first edit: item 1 is still merely proposed there
    answers[0]({
      session_id: 's1',
      assignment: [
        { item: 0, candidate: 0, left: 'a', right: 'a', score: 1, status: 'confirmed' },
        { item: 1, candidate: 1, left: 'b', right: 'b', score: 1, status: 'proposed' },
      ],
      changed: [],
      accounting: { tiers: {} },
    });
    await first;
    const status = (i: number) => store.getState().session?.review.find((r) => r.item === i)?.status;
    expect(status(0)).toBe('confirmed');
    expect(status(1)).toBe('confirmed'); // not flipped back while its own edit is in flight
  });

  it('labels through the Oracle endpoint and takes its next query', async () => {
    const next = { item: 1, left: 'b', candidate: { candidate: 1, right: 'b', score: 1, explanation: null }, margin: 0.3 };
    const client = fakeClient({
      next: vi.fn(async () => ({ ...next, item: 0, candidate: { ...next.candidate, candidate: 0 } })),
      label: vi.fn(async () => ({
        label: { item: 0, candidate: 0, is_match: true, source: 'human' },
        result: { session_id: 's1', assignment: session.assignment, changed: [], accounting: { tiers: {} } },
        next,
      })) as Client['label'],
    });
    const store = createReviewStore(client);
    await store.getState().open({ A: [], B: [] });
    await store.getState().label(true);
    expect(client.label).toHaveBeenCalledWith('s1', 0, 0, true);
    expect(client.edit).not.toHaveBeenCalled();
    expect(store.getState().query?.item).toBe(1);
    expect(store.getState().session?.labels).toHaveLength(1);
  });
});
