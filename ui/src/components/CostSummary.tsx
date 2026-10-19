/** The session's per-tier cost/latency accounting, as a compact line. */
import type { Accounting } from '../schema/equate';

export function CostSummary({ accounting }: { accounting: Accounting }) {
  const tiers = Object.entries(accounting.tiers);
  if (!tiers.length) return null;
  return (
    <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground" aria-label="Time spent per stage">
      {tiers.map(([tier, c]) => (
        <div key={tier} className="flex gap-1">
          <dt>{tier}</dt>
          <dd className="tabular-nums">
            {c.calls}× · {(c.seconds * 1000).toFixed(1)} ms
          </dd>
        </div>
      ))}
    </dl>
  );
}
