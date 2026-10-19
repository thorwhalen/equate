/**
 * Turning pasted text into the items of a collection, and items back into display text.
 *
 * Accepted input: a JSON array (of strings or records), JSON lines (one record per line),
 * or plain lines (one item per line). Blank lines are ignored.
 */

export function parseItems(text: string): unknown[] {
  const t = text.trim();
  if (!t) return [];
  if (t.startsWith('[')) {
    const parsed: unknown = JSON.parse(t);
    if (!Array.isArray(parsed)) throw new Error('expected a JSON array');
    return parsed;
  }
  const lines = t.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines.every((l) => l.startsWith('{'))) return lines.map((l) => JSON.parse(l));
  return lines;
}

/** A one-line rendering of an item: a record's non-empty values joined, else the value. */
export function displayText(obj: unknown): string {
  if (obj === null || obj === undefined) return '';
  if (typeof obj === 'object' && !Array.isArray(obj)) {
    return Object.values(obj as Record<string, unknown>)
      .filter((v) => v !== null && v !== undefined && v !== '')
      .map(String)
      .join(' · ');
  }
  return String(obj);
}

export function formatScore(x: number | null | undefined, digits = 3): string {
  return x === null || x === undefined ? '—' : x.toFixed(digits);
}
