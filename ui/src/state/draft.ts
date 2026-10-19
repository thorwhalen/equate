/**
 * The setup form's draft, behind a zodal `DataProvider` (frontend invariant 4): the
 * pasted collections survive a reload, and the provider can be swapped for a server-side
 * one without touching the form. Cleared only after a session opened successfully.
 */
import { createLocalStorageProvider } from '@zodal/store-localstorage';
import type { DataProvider } from '@zodal/store';

export interface Draft {
  id: string;
  left: string;
  right: string;
  k: string; // as typed: the draft is not the value (validated on submit)
  threshold?: string; // absent in drafts saved before equate#21
  compare: string;
  savedAt: number;
}

const DRAFT_ID = 'setup';

export function createDraftStore(
  provider: DataProvider<Draft> = createLocalStorageProvider<Draft>({
    storageKey: 'equate-review:draft',
    idField: 'id',
  }),
) {
  return {
    async load(): Promise<Draft | null> {
      try {
        return await provider.getOne(DRAFT_ID);
      } catch {
        return null; // none yet, or storage unavailable (private window)
      }
    },
    async save(d: Omit<Draft, 'id' | 'savedAt'>): Promise<void> {
      const draft: Draft = { ...d, id: DRAFT_ID, savedAt: Date.now() };
      try {
        await provider.update(DRAFT_ID, draft);
      } catch {
        try {
          await provider.create(draft);
        } catch {
          /* storage unavailable: the form still works, it just won't survive a reload */
        }
      }
    },
    async clear(): Promise<void> {
      try {
        await provider.delete(DRAFT_ID);
      } catch {
        /* nothing to clear */
      }
    },
  };
}

export type DraftStore = ReturnType<typeof createDraftStore>;
