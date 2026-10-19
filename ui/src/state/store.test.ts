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
    const done = store.getState().edit({ action: 'confirm', item: 0, candidate: 1 });
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
    await store.getState().edit({ action: 'reject', item: 0, candidate: 0 });
    expect(store.getState().error).toMatch(/not applied: nope/);
    expect(client.getSession).toHaveBeenCalledWith('s1');
    expect(store.getState().session?.review[0].status).toBe('proposed');
  });
});
