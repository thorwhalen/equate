/** A side-by-side field diff: both values per field, with their similarity. */
import type { CandidateView } from '../schema/equate';
import { fieldDiffs } from '../schema/equate';
import { cn } from '../util/cn';

const show = (v: unknown) => (v === null || v === undefined || v === '' ? '—' : String(v));

export function FieldDiffTable({ candidate }: { candidate: CandidateView }) {
  const diffs = fieldDiffs(candidate);
  return (
    <table className="w-full table-fixed text-xs">
      <thead className="text-left text-muted-foreground">
        <tr>
          <th scope="col" className="w-24 py-1 font-medium">Field</th>
          <th scope="col" className="py-1 font-medium">Item</th>
          <th scope="col" className="py-1 font-medium">Candidate</th>
          <th scope="col" className="w-20 py-1 text-right font-medium">Similarity</th>
        </tr>
      </thead>
      <tbody>
        {diffs.map((d) => {
          const differs = d.similarity === null || d.similarity < 1;
          return (
            <tr key={d.field} className="border-t border-border align-top">
              <th scope="row" className="py-1 pr-2 text-left font-medium">{d.field}</th>
              <td className="break-words py-1 pr-2">{show(d.left)}</td>
              <td className={cn('break-words py-1 pr-2', differs && 'font-semibold')}>{show(d.right)}</td>
              <td className="py-1 text-right tabular-nums">
                {d.similarity === null ? 'missing' : d.similarity.toFixed(2)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
