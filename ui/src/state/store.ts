/**
 * The review session store (zustand + immer). One store for the one surface.
 *
 * Edits are optimistic: the predicted effect shows at once (`applyOptimistic`), the edit
 * is sent, and the server's locally re-optimized assignment replaces the prediction
 * (`reconcile`); the items it changed are flashed. Edits are sent one at a time, in order,
 * because each is a constraint on top of the previous ones. On a failed edit the session
 * is re-read from the server, which is the authority.
 */
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { Client, OpenSessionParams } from '../api/client';
import type { Edit, EditResult, Query, SessionView } from '../schema/equate';
import { applyOptimistic, reconcile } from './optimistic';

export type Status = 'idle' | 'loading' | 'ready' | 'error';

export interface ReviewState {
  session: SessionView | null;
  selected: number | null;
  pending: number[];
  flash: number[];
  query: Query | null;
  status: Status;
  error: string | null;
  open: (p: OpenSessionParams) => Promise<string | null>;
  load: (sessionId: string, force?: boolean) => Promise<void>;
  reset: () => void;
  select: (item: number | null) => void;
  edit: (edit: Edit) => Promise<void>;
  label: (isMatch: boolean) => Promise<void>;
  moreCandidates: (item: number) => Promise<void>;
  dismissError: () => void;
}

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function createReviewStore(client: Client) {
  let chain: Promise<void> = Promise.resolve();
  // edits sent but not yet answered, oldest first: a reconcile reflects only the edits the
  // server has answered, so these are re-applied on top or the screen would flip back
  let unacked: Edit[] = [];

  return create<ReviewState>()(
    immer((set, get) => {
      function landed(id: string, edit: Edit, result: EditResult) {
        unacked = unacked.filter((e) => e !== edit);
        set((s) => {
          if (s.session?.session_id !== id) return;
          let next = reconcile(s.session, result);
          for (const e of unacked) next = applyOptimistic(next, e, { record: false });
          s.session = next;
          s.flash = result.changed;
        });
      }

      /** Show `edit` at once, then run `request` after every earlier edit has landed. */
      function send(edit: Edit, request: (id: string) => Promise<void>): Promise<void> {
        const session = get().session;
        if (!session) return Promise.resolve();
        const id = session.session_id;
        set((s) => {
          s.session = applyOptimistic(s.session as SessionView, edit);
          s.pending.push(edit.item);
          s.flash = [];
        });
        unacked.push(edit);
        chain = chain.then(async () => {
          try {
            await request(id);
          } catch (e) {
            unacked = [];
            await get().load(id, true); // the server is the authority: drop the prediction
            set((s) => {
              s.error = `That edit was not applied: ${message(e)}`;
            });
          } finally {
            set((s) => {
              const at = s.pending.indexOf(edit.item);
              if (at >= 0) s.pending.splice(at, 1);
            });
          }
        });
        return chain;
      }

      async function refreshQuery(id: string) {
        const query = await client.next(id);
        set((s) => {
          s.query = query;
        });
      }

      return {
        session: null,
        selected: null,
        pending: [],
        flash: [],
        query: null,
        status: 'idle',
        error: null,

        async open(p) {
          set((s) => {
            s.status = 'loading';
            s.session = null; // never leave the previous session on screen while loading
            s.error = null;
          });
          try {
            const session = await client.openSession(p);
            set((s) => {
              s.session = session;
              s.selected = session.review[0]?.item ?? null;
              s.status = 'ready';
            });
            await refreshQuery(session.session_id);
            return session.session_id;
          } catch (e) {
            set((s) => {
              s.status = 'error';
              s.error = message(e);
            });
            return null;
          }
        },

        async load(sessionId, force = false) {
          const current = get();
          if (!force && current.session?.session_id === sessionId && current.status === 'ready')
            return;
          set((s) => {
            s.status = 'loading';
            s.session = null;
            s.error = null;
          });
          try {
            const session = await client.getSession(sessionId);
            set((s) => {
              s.session = session;
              s.selected = session.review[0]?.item ?? null;
              s.pending = [];
              s.status = 'ready';
            });
            await refreshQuery(sessionId);
          } catch (e) {
            set((s) => {
              s.status = 'error';
              s.error = message(e);
            });
          }
        },

        reset() {
          set((s) => {
            s.session = null;
            s.selected = null;
            s.query = null;
            s.status = 'idle';
            s.error = null;
          });
        },

        select(item) {
          set((s) => {
            s.selected = item;
          });
        },

        edit(edit) {
          return send(edit, (id) =>
            client.edit(id, [edit]).then(async (result) => {
              landed(id, edit, result);
              await refreshQuery(id);
            }),
          );
        },

        async label(isMatch) {
          const { session, query } = get();
          if (!session || !query) return;
          const edit: Edit = {
            action: isMatch ? 'confirm' : 'reject',
            item: query.item,
            candidate: query.candidate.candidate,
            source: 'human',
          };
          // the Oracle endpoint: records the label with its edit, and returns the next query
          await send(edit, (id) =>
            client.label(id, edit.item, edit.candidate, isMatch).then((res) => {
              landed(id, edit, res.result);
              set((s) => {
                s.session?.labels.push(res.label);
                s.query = res.next;
              });
            }),
          );
        },

        async moreCandidates(item) {
          const session = get().session;
          if (!session) return;
          const row = session.review.find((r) => r.item === item);
          const k = (row?.candidates.length ?? session.k) + session.k;
          try {
            const candidates = await client.candidates(session.session_id, item, k);
            set((s) => {
              const r = s.session?.review.find((x) => x.item === item);
              if (r) r.candidates = candidates;
            });
          } catch (e) {
            set((s) => {
              s.error = message(e);
            });
          }
        },

        dismissError() {
          set((s) => {
            s.error = null;
          });
        },
      };
    }),
  );
}

export type ReviewStore = ReturnType<typeof createReviewStore>;
