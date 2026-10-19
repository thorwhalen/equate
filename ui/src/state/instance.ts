/** The app's single client, review store and draft store (the wiring point for tests). */
import { createClient } from '../api/client';
import { createDraftStore } from './draft';
import { createReviewStore } from './store';

export const client = createClient();
export const useReview = createReviewStore(client);
export const drafts = createDraftStore();
