/**
 * The review queue, rendered from the zodal collection: columns come from `toColumnDefs`,
 * each cell from the shadcn renderer registry. Rows are ordered most-uncertain first by
 * the server; the rows a re-optimization changed flash, rows with an edit in flight dim.
 */
import type { ComponentType } from 'react';
import { registry, reviewCollection, reviewColumns, toRows, type ReviewRow } from '../review/collection';
import { useReview } from '../state/instance';
import { cn } from '../util/cn';
import { Badge, type BadgeVariant } from './ui/badge';

type CellComponent = ComponentType<{ value: unknown; config: unknown; row: Record<string, unknown> }>;

const STATUS_VARIANT: Record<ReviewRow['status'], BadgeVariant> = {
  proposed: 'secondary',
  confirmed: 'default',
  unmatched: 'outline',
};

function Cell({ column, row }: { column: (typeof reviewColumns)[number]; row: ReviewRow }) {
  const value = row[column.id as keyof ReviewRow];
  if (column.id === 'status') {
    return <Badge variant={STATUS_VARIANT[row.status]}>{row.status}</Badge>;
  }
  if ((column.id === 'score' || column.id === 'margin') && typeof value === 'number') {
    return <span className="tabular-nums">{value.toFixed(3)}</span>;
  }
  const field = reviewCollection.fieldAffordances[column.id];
  const Renderer = field
    ? (registry.resolve(field, { mode: 'cell' }) as CellComponent | null)
    : null;
  if (Renderer) return <Renderer value={value ?? ''} config={column} row={row} />;
  return <span>{value === null || value === undefined ? '—' : String(value)}</span>;
}

export function ReviewTable() {
  const session = useReview((s) => s.session);
  const selected = useReview((s) => s.selected);
  const pending = useReview((s) => s.pending);
  const flash = useReview((s) => s.flash);
  const select = useReview((s) => s.select);
  if (!session) return null;
  const rows = toRows(session);

  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full text-sm">
        <caption className="sr-only">Review queue, most uncertain first</caption>
        <thead className="bg-muted text-left text-xs text-muted-foreground">
          <tr>
            {reviewColumns.map((c) => (
              <th key={c.id} scope="col" className="px-3 py-2 font-medium">
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={`${row.item}-${flash.includes(row.item) ? 'f' : ''}`}
              tabIndex={0}
              aria-selected={row.item === selected}
              aria-busy={pending.includes(row.item)}
              onClick={() => select(row.item)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  select(row.item);
                }
              }}
              className={cn(
                'cursor-pointer border-t border-border outline-none hover:bg-muted/60 focus-visible:ring-2 focus-visible:ring-ring',
                row.item === selected && 'bg-muted',
                flash.includes(row.item) && 'row-flash',
                pending.includes(row.item) && 'opacity-60',
              )}
            >
              {reviewColumns.map((c) => (
                <td
                  key={c.id}
                  className={cn(
                    'px-3 py-2',
                    c.id === 'left' || c.id === 'match' ? 'max-w-[14rem] truncate' : 'whitespace-nowrap',
                  )}
                >
                  <Cell column={c} row={row} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
