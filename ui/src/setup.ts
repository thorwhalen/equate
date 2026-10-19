/**
 * The setup form's numeric parameters: what the person typed (a draft string) and what it
 * means (a valid value), kept apart (tw-frontend-ux controls: "the draft is not the
 * value"). The Zod schema is the one rule; the form shows its message inline and never
 * rewrites the text being typed.
 */
import { z } from 'zod';

export const DEFAULT_K = 5;
export const MAX_K = 50;

/** A plain decimal number, as Python's float() reads it (no hex/binary/octal, no "Infinity"). */
const DECIMAL = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

/** A typed decimal: ``null`` when blank, a number when it is one, an issue otherwise. */
function decimal(label: string, { required }: { required: boolean }) {
  return z.string().transform((raw, ctx) => {
    const t = raw.trim();
    if (!t) {
      if (required) ctx.addIssue({ code: 'custom', message: `${label} is required` });
      return required ? z.NEVER : null;
    }
    const n = Number(t);
    if (!DECIMAL.test(t) || !Number.isFinite(n)) {
      ctx.addIssue({ code: 'custom', message: `${label} must be a decimal number` });
      return z.NEVER;
    }
    return n;
  });
}

export const SetupParams = z.object({
  k: decimal('Candidates per item', { required: true }).pipe(
    z
      .number()
      .int('Candidates per item must be a whole number')
      .min(1, 'Candidates per item must be at least 1')
      .max(MAX_K, `Candidates per item must be at most ${MAX_K}`),
  ),
  // the lowest acceptable score; blank means "no floor"
  floor: decimal('Minimum score', { required: false }),
});
export type SetupParams = z.infer<typeof SetupParams>;

export type FieldErrors = Partial<Record<keyof SetupParams, string>>;

/** Parse the typed `k` and `floor` drafts: the values, or one message per bad field. */
export function parseSetup(draft: { k: string; floor: string }):
  | { ok: true; value: SetupParams }
  | { ok: false; errors: FieldErrors } {
  const r = SetupParams.safeParse(draft);
  if (r.success) return { ok: true, value: r.data };
  const errors: FieldErrors = {};
  for (const issue of r.error.issues) {
    const key = issue.path[0] as keyof SetupParams;
    errors[key] ??= issue.message;
  }
  return { ok: false, errors };
}
