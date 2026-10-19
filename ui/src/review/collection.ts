/**
 * The review queue as a zodal collection: a Zod schema for one queue row, from which the
 * table's columns (and their cell renderers, via the shadcn registry) are generated.
 */
import { z } from 'zod';
import { defineCollection } from '@zodal/core';
import { toColumnDefs } from '@zodal/ui';
import { createShadcnRegistry } from '@zodal/ui-shadcn';
import type { SessionView } from '../schema/equate';
import { displayText } from '../util/items';

export const ReviewRow = z.object({
  item: z.number().int().meta({ title: '#' }),
  left: z.string().meta({ title: 'Item' }),
  match: z.string().meta({ title: 'Current match' }),
  score: z.number().nullable().meta({ title: 'Score' }),
  margin: z.number().nullable().meta({ title: 'Margin' }),
  status: z.enum(['proposed', 'confirmed', 'unmatched']).meta({ title: 'Status' }),
});
export type ReviewRow = z.infer<typeof ReviewRow>;

export const reviewCollection = defineCollection(ReviewRow, {
  idField: 'item',
  affordances: { selectable: false }, // one row is focused at a time; no bulk selection
  labelField: 'left',
  fields: {
    status: { badge: { proposed: 'secondary', confirmed: 'default', unmatched: 'outline' } },
  },
});

export const reviewColumns = toColumnDefs(reviewCollection);
export const registry = createShadcnRegistry();

/** Flatten a session's review queue into table rows (queue order kept). */
export function toRows(session: SessionView): ReviewRow[] {
  return session.review.map((r) => {
    const c = r.candidates.find((x) => x.candidate === r.assigned);
    const pair = session.assignment.find((p) => p.item === r.item);
    return {
      item: r.item,
      left: displayText(r.left),
      match: r.assigned === null ? '' : displayText(c?.right ?? pair?.right),
      score: c?.score ?? pair?.score ?? null,
      margin: r.margin,
      status: r.status,
    };
  });
}
