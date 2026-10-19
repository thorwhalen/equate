/**
 * The active-learning labelling widget (the Oracle UI): the pair whose label would teach
 * the most — the most uncertain item, and its current match — with a yes/no answer.
 */
import { useReview } from '../state/instance';
import { displayText, formatScore } from '../util/items';
import { FieldDiffTable } from './FieldDiffTable';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';

export function LabelWidget() {
  const query = useReview((s) => s.query);
  const pending = useReview((s) => s.pending);
  const label = useReview((s) => s.label);
  const select = useReview((s) => s.select);
  const busy = pending.length > 0;

  return (
    <Card aria-busy={busy}>
      <CardHeader>
        <CardTitle>Is this a match?</CardTitle>
        <CardDescription>
          {query
            ? `The least certain pair (margin ${formatScore(query.margin)}). Your answer is applied and the matching re-solved.`
            : 'Nothing left to label: every item is confirmed or has no open candidate.'}
        </CardDescription>
      </CardHeader>
      {query && (
        <CardContent className="flex flex-col gap-3">
          <p className="text-sm">
            <button className="font-medium underline-offset-2 hover:underline" onClick={() => select(query.item)}>
              {displayText(query.left)}
            </button>
            <span className="text-muted-foreground"> vs </span>
            <span className="font-medium">{displayText(query.candidate.right)}</span>
          </p>
          <FieldDiffTable candidate={query.candidate} />
          <div className="flex gap-2">
            <Button disabled={busy} onClick={() => label(true)}>
              Match
            </Button>
            <Button variant="outline" disabled={busy} onClick={() => label(false)}>
              Not a match
            </Button>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
