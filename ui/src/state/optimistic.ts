/**
 * Optimistic edits: what the UI shows between sending an edit and the server's answer.
 *
 * Pure functions over a `SessionView`, so they are tested without React. `applyOptimistic`
 * predicts the edit's direct effect (the confirmed pair, the displaced item); `reconcile`
 * replaces the prediction with the server's locally re-optimized assignment, keeping the
 * review queue's order (rows do not jump under the reviewer's cursor).
 */
import { produce } from 'immer';
import type { Edit, EditResult, PairView, ReviewItem, SessionView } from '../schema/equate';

function candidateOf(row: ReviewItem | undefined, j: number) {
  return row?.candidates.find((c) => c.candidate === j);
}

export function applyOptimistic(session: SessionView, edit: Edit): SessionView {
  return produce(session, (d) => {
    const row = d.review.find((r) => r.item === edit.item);
    if (edit.action === 'confirm' || edit.action === 'reassign') {
      const displaced = d.assignment.filter(
        (p) => p.candidate === edit.candidate && p.item !== edit.item,
      );
      d.assignment = d.assignment.filter(
        (p) => p.item !== edit.item && p.candidate !== edit.candidate,
      );
      const c = candidateOf(row, edit.candidate);
      d.assignment.push({
        item: edit.item,
        candidate: edit.candidate,
        left: row?.left,
        right: c?.right,
        score: c?.score ?? null,
        status: 'confirmed',
      });
      if (row) {
        row.assigned = edit.candidate;
        row.status = 'confirmed';
      }
      for (const p of displaced) {
        const other = d.review.find((r) => r.item === p.item);
        if (other) {
          other.assigned = null;
          other.status = 'unmatched';
        }
      }
    } else if (edit.action === 'reject') {
      d.assignment = d.assignment.filter(
        (p) => !(p.item === edit.item && p.candidate === edit.candidate),
      );
      if (row && row.assigned === edit.candidate) {
        row.assigned = null;
        row.status = 'unmatched';
      }
    } else if (edit.action === 'clear') {
      const p = d.assignment.find((q) => q.item === edit.item && q.candidate === edit.candidate);
      if (p) p.status = 'proposed';
      if (row && row.assigned === edit.candidate) row.status = 'proposed';
    }
    d.edits.push(edit);
  });
}

/** Statuses and assignment from the server's answer; queue order is kept. */
export function reconcile(session: SessionView, result: EditResult): SessionView {
  return produce(session, (d) => {
    d.assignment = result.assignment;
    d.accounting = result.accounting;
    const byItem = new Map<number, PairView>(result.assignment.map((p) => [p.item, p]));
    for (const row of d.review) {
      const p = byItem.get(row.item);
      row.assigned = p ? p.candidate : null;
      row.status = p ? p.status : 'unmatched';
    }
  });
}

/** The pairs currently rejected, folding the edit log as the server does (latest wins). */
export function rejectedPairs(edits: Edit[]): Set<string> {
  const rejected = new Set<string>();
  for (const e of edits) {
    const key = `${e.item}:${e.candidate}`;
    if (e.action === 'reject') rejected.add(key);
    else rejected.delete(key); // confirm / reassign / clear lift a reject
  }
  return rejected;
}
