/**
 * Zod schemas for the equate service's JSON boundary — the UI's single source of truth.
 *
 * Each schema mirrors a Python dataclass in `equate/service.py` (and `Explanation` in
 * `equate/base.py`). Python is the source; `fields.json` is its field-name snapshot
 * (`python -m equate.service fields`), and `equate.test.ts` fails when a schema here and
 * the snapshot disagree, so the two languages cannot drift apart silently.
 */
import { z } from 'zod';

const index = z.number().int().nonnegative();
/** A left/right object: a string, a number, or a JSON record. */
const obj = z.unknown();

export const EditAction = z.enum(['confirm', 'reject', 'reassign', 'clear']);

export const Edit = z.object({
  action: EditAction,
  item: index,
  candidate: index,
  source: z.string(), // who made it: 'human', 'llm', ...
});

export const Label = z.object({
  item: index,
  candidate: index,
  is_match: z.boolean(),
  source: z.string(),
});

export const FieldDiff = z.object({
  field: z.string(),
  left: obj,
  right: obj,
  similarity: z.number().nullable(),
});

export const Explanation = z.object({
  summary: z.string(),
  details: z.record(z.string(), z.unknown()),
});

export const CandidateView = z.object({
  candidate: index,
  right: obj,
  score: z.number().nullable(),
  explanation: Explanation.nullable(),
});

export const PairStatus = z.enum(['proposed', 'confirmed']);

export const PairView = z.object({
  item: index,
  candidate: index,
  left: obj,
  right: obj,
  score: z.number().nullable(),
  status: PairStatus,
});

export const ItemStatus = z.enum(['proposed', 'confirmed', 'unmatched']);

export const ReviewItem = z.object({
  item: index,
  left: obj,
  assigned: index.nullable(),
  status: ItemStatus,
  margin: z.number().nullable(),
  candidates: z.array(CandidateView),
});

export const Query = z.object({
  item: index,
  left: obj,
  candidate: CandidateView,
  margin: z.number().nullable(),
});

export const TierCost = z.object({ calls: z.number(), seconds: z.number(), items: z.number() });

export const Accounting = z.object({ tiers: z.record(z.string(), TierCost) });

export const MatchResult = z.object({
  pairs: z.array(PairView),
  groups: z.array(z.array(index)).nullable(),
  accounting: Accounting,
  cached: z.boolean(),
});

export const SessionView = z.object({
  session_id: z.string(),
  n_left: z.number().int(),
  n_right: z.number().int(),
  sense: z.enum(['maximize', 'minimize']),
  k: z.number().int(),
  assignment: z.array(PairView),
  review: z.array(ReviewItem),
  edits: z.array(Edit),
  labels: z.array(Label),
  accounting: Accounting,
  threshold: z.number().nullable(), // the score floor (a ceiling for 'minimize')
});

export const EditResult = z.object({
  session_id: z.string(),
  assignment: z.array(PairView),
  changed: z.array(index),
  accounting: Accounting,
});

export const LabelResult = z.object({
  label: Label,
  result: EditResult,
  next: Query.nullable(),
});

/** Every boundary schema by its Python type name (checked against `fields.json`). */
export const BOUNDARY = {
  Edit,
  Label,
  FieldDiff,
  Explanation,
  CandidateView,
  PairView,
  ReviewItem,
  Query,
  TierCost,
  Accounting,
  MatchResult,
  SessionView,
  EditResult,
  LabelResult,
} as const;

export type Edit = z.infer<typeof Edit>;
export type EditAction = z.infer<typeof EditAction>;
export type Label = z.infer<typeof Label>;
export type FieldDiff = z.infer<typeof FieldDiff>;
export type Explanation = z.infer<typeof Explanation>;
export type CandidateView = z.infer<typeof CandidateView>;
export type PairView = z.infer<typeof PairView>;
export type ReviewItem = z.infer<typeof ReviewItem>;
export type Query = z.infer<typeof Query>;
export type Accounting = z.infer<typeof Accounting>;
export type MatchResult = z.infer<typeof MatchResult>;
export type SessionView = z.infer<typeof SessionView>;
export type EditResult = z.infer<typeof EditResult>;
export type LabelResult = z.infer<typeof LabelResult>;

/** The field diffs carried in a candidate's explanation (empty when there are none). */
export function fieldDiffs(c: CandidateView): FieldDiff[] {
  const parsed = z.array(FieldDiff).safeParse(c.explanation?.details?.fields);
  return parsed.success ? parsed.data : [];
}
