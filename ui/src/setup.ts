/**
 * The setup form's numeric parameters: what the person typed (a draft string) and what it
 * means (a valid value), kept apart (tw-frontend-ux controls: "the draft is not the
 * value"). The Zod schema is the one rule; the form shows its message inline and never
 * rewrites the text being typed.
 */
import { z } from 'zod';

export const DEFAULT_K = 5;
export const MAX_K = 50;

export const SetupParams = z.object({
  k: z.coerce
    .number({ message: 'Candidates per item must be a number' })
    .int('Candidates per item must be a whole number')
    .min(1, 'Candidates per item must be at least 1')
    .max(MAX_K, `Candidates per item must be at most ${MAX_K}`),
  // empty means "no floor"
  threshold: z.string().transform((raw, ctx) => {
    const t = raw.trim();
    if (!t) return null;
    const n = Number(t);
    if (!Number.isFinite(n)) {
      ctx.addIssue({ code: 'custom', message: 'Minimum score must be a number (or empty)' });
      return z.NEVER;
    }
    return n;
  }),
});
export type SetupParams = z.infer<typeof SetupParams>;

export type FieldErrors = Partial<Record<keyof SetupParams, string>>;

/** Parse the typed `k` and `threshold` drafts: the values, or one message per bad field. */
export function parseSetup(draft: { k: string; threshold: string }):
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
