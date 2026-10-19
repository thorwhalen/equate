/**
 * The selected item and its candidates, each with a field diff and the edit actions:
 * confirm the current match, use another candidate (reassign), reject, or undo.
 */
import { useReview } from '../state/instance';
import { rejectedPairs } from '../state/optimistic';
import type { CandidateView, EditAction } from '../schema/equate';
import { displayText, formatScore } from '../util/items';
import { FieldDiffTable } from './FieldDiffTable';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';

export function PairDetail() {
  const session = useReview((s) => s.session);
  const selected = useReview((s) => s.selected);
  const pending = useReview((s) => s.pending);
  const edit = useReview((s) => s.edit);
  const more = useReview((s) => s.moreCandidates);
  const row = session?.review.find((r) => r.item === selected);
  if (!session || !row) {
    return (
      <Card>
        <CardContent className="p-6 text-sm text-muted-foreground">
          Select an item in the queue to review its candidates.
        </CardContent>
      </Card>
    );
  }
  const rejected = rejectedPairs(session.edits);
  const busy = pending.includes(row.item);
  const act = (action: EditAction, c: CandidateView) =>
    edit({ action, item: row.item, candidate: c.candidate });

  return (
    <Card aria-busy={busy}>
      <CardHeader>
        <CardTitle>
          Item #{row.item}: {displayText(row.left)}
        </CardTitle>
        <CardDescription>
          {row.status === 'unmatched'
            ? 'No match proposed.'
            : row.status === 'confirmed'
              ? 'You confirmed this match.'
              : `Proposed match, margin ${formatScore(row.margin)} over the runner-up.`}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {row.candidates.map((c, rank) => {
          const isCurrent = row.assigned === c.candidate;
          const isRejected = rejected.has(`${row.item}:${c.candidate}`);
          return (
            <section
              key={c.candidate}
              aria-label={`Candidate ${rank + 1}`}
              className="rounded-lg border border-border p-3"
            >
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium">{displayText(c.right)}</span>
                <span className="text-xs tabular-nums text-muted-foreground">
                  score {formatScore(c.score)}
                </span>
                {isCurrent && (
                  <Badge variant={row.status === 'confirmed' ? 'default' : 'secondary'}>
                    {row.status === 'confirmed' ? 'confirmed' : 'current match'}
                  </Badge>
                )}
                {isRejected && <Badge variant="outline">rejected</Badge>}
                <span className="ml-auto flex gap-2">
                  {isRejected ? (
                    <Button size="sm" variant="ghost" disabled={busy} onClick={() => act('clear', c)}>
                      Undo reject
                    </Button>
                  ) : isCurrent && row.status === 'confirmed' ? (
                    <Button size="sm" variant="ghost" disabled={busy} onClick={() => act('clear', c)}>
                      Undo confirm
                    </Button>
                  ) : (
                    <>
                      <Button size="sm" disabled={busy} onClick={() => act(isCurrent ? 'confirm' : 'reassign', c)}>
                        {isCurrent ? 'Confirm' : 'Use this match'}
                      </Button>
                      <Button size="sm" variant="outline" disabled={busy} onClick={() => act('reject', c)}>
                        Reject
                      </Button>
                    </>
                  )}
                </span>
              </div>
              <FieldDiffTable candidate={c} />
            </section>
          );
        })}
        {row.candidates.length < session.n_right && (
          <Button variant="ghost" size="sm" className="self-start" onClick={() => more(row.item)}>
            Show more candidates
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
