/**
 * The setup screen: paste two collections, choose how to compare them, open a session.
 * The draft autosaves (debounced) and is restored on return; it is cleared only after a
 * session opened successfully (tw-frontend-ux principle 4).
 */
import { useEffect, useRef, useState } from 'react';
import { DEMO_LEFT, DEMO_RIGHT, toJsonLines } from '../demo';
import { drafts, useReview } from '../state/instance';
import { navigate } from '../route';
import { parseItems } from '../util/items';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';

const COMPARATORS = [
  { value: '', label: 'TF-IDF + cosine (default)' },
  { value: 'ratio', label: 'Edit ratio (difflib)' },
  { value: 'jaro_winkler', label: 'Jaro-Winkler' },
  { value: 'monge_elkan', label: 'Monge-Elkan (token-wise)' },
];
const DEFAULT_K = 5;
const SAVE_DELAY_MS = 400;

function count(text: string): string {
  try {
    return `${parseItems(text).length} items`;
  } catch {
    return 'not valid JSON yet';
  }
}

export function SetupForm() {
  const open = useReview((s) => s.open);
  const status = useReview((s) => s.status);
  const [left, setLeft] = useState('');
  const [right, setRight] = useState('');
  const [k, setK] = useState(DEFAULT_K);
  const [compare, setCompare] = useState('');
  const [restoredAt, setRestoredAt] = useState<number | null>(null);
  const [inputError, setInputError] = useState<string | null>(null);
  const loaded = useRef(false);

  useEffect(() => {
    drafts.load().then((d) => {
      if (d && !loaded.current && (d.left || d.right)) {
        setLeft(d.left);
        setRight(d.right);
        setK(d.k);
        setCompare(d.compare);
        setRestoredAt(d.savedAt);
      }
      loaded.current = true;
    });
  }, []);

  useEffect(() => {
    if (!loaded.current) return;
    const t = setTimeout(() => drafts.save({ left, right, k, compare }), SAVE_DELAY_MS);
    return () => clearTimeout(t);
  }, [left, right, k, compare]);

  async function start() {
    let A: unknown[];
    let B: unknown[];
    try {
      A = parseItems(left);
      B = parseItems(right);
    } catch (e) {
      setInputError(`Could not read the items: ${(e as Error).message}`);
      return;
    }
    if (!A.length || !B.length) {
      setInputError('Both collections need at least one item.');
      return;
    }
    setInputError(null);
    const id = await open({ A, B, k, compare: compare || null });
    if (id) {
      await drafts.clear();
      navigate({ session: id });
    }
  }

  const busy = status === 'loading';
  return (
    <Card className="mx-auto max-w-5xl">
      <CardHeader>
        <CardTitle>New review session</CardTitle>
        <CardDescription>
          Paste two collections: one item per line, JSON lines, or a JSON array. equate
          scores every pair, proposes the best one-to-one matching, and lets you correct it.
        </CardDescription>
        {restoredAt && (
          <p className="text-xs text-muted-foreground" role="status">
            Restored your draft from {new Date(restoredAt).toLocaleString()}.
          </p>
        )}
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="grid gap-4 md:grid-cols-2">
          {(
            [
              ['left', 'Left collection', left, setLeft],
              ['right', 'Right collection', right, setRight],
            ] as const
          ).map(([id, label, value, setValue]) => (
            <label key={id} className="flex flex-col gap-1 text-sm font-medium">
              <span className="flex justify-between">
                {label}
                <span className="font-normal text-muted-foreground">{value && count(value)}</span>
              </span>
              <textarea
                className="min-h-48 max-h-96 resize-y rounded-md border border-border bg-background p-2 font-mono text-xs"
                value={value}
                onChange={(e) => setValue(e.target.value)}
                spellCheck={false}
              />
            </label>
          ))}
        </div>
        <div className="flex flex-wrap items-end gap-4">
          <label className="flex flex-col gap-1 text-sm font-medium">
            Compare with
            <select
              className="h-9 rounded-md border border-border bg-background px-2 text-sm"
              value={compare}
              onChange={(e) => setCompare(e.target.value)}
            >
              {COMPARATORS.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-sm font-medium">
            Candidates per item
            <input
              type="number"
              min={1}
              max={50}
              className="h-9 w-24 rounded-md border border-border bg-background px-2 text-sm"
              value={k}
              onChange={(e) => setK(Math.max(1, Number(e.target.value) || DEFAULT_K))}
            />
          </label>
          <div className="ml-auto flex gap-2">
            <Button
              variant="outline"
              onClick={() => {
                setLeft(toJsonLines(DEMO_LEFT));
                setRight(toJsonLines(DEMO_RIGHT));
              }}
            >
              Load demo data
            </Button>
            <Button onClick={start} disabled={busy} aria-busy={busy}>
              {busy ? 'Scoring…' : 'Start review'}
            </Button>
          </div>
        </div>
        {inputError && (
          <p className="text-sm text-destructive" role="alert">
            {inputError}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
