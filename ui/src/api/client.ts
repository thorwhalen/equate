/**
 * The equate service client. Every response is parsed with its Zod schema, so a payload
 * that drifts from the contract fails loudly here, not three components later.
 *
 * `base` is the seam: `/api` (the vite dev proxy) by default, `VITE_EQUATE_API` at build
 * time, or any URL passed to `createClient`.
 */
import type { z } from 'zod';
import {
  CandidateView,
  EditResult,
  LabelResult,
  Query,
  SessionView,
  type Edit,
} from '../schema/equate';

export const DEFAULT_API_BASE: string = import.meta.env?.VITE_EQUATE_API ?? '/api';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export interface OpenSessionParams {
  A: unknown[];
  B: unknown[];
  k?: number;
  compare?: string | null;
  featurize?: string | null;
  threshold?: number | null;
}

export function createClient(base: string = DEFAULT_API_BASE, fetchFn: typeof fetch = fetch) {
  async function call<S extends z.ZodTypeAny>(
    schema: S,
    path: string,
    init?: { method?: string; body?: unknown },
  ): Promise<z.infer<S>> {
    const res = await fetchFn(`${base}${path}`, {
      method: init?.method ?? 'GET',
      headers: init?.body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: init?.body !== undefined ? JSON.stringify(init.body) : undefined,
    });
    const text = await res.text();
    const data = text ? JSON.parse(text) : null;
    if (!res.ok) {
      const detail = data && typeof data.detail === 'string' ? data.detail : res.statusText;
      throw new ApiError(detail || `HTTP ${res.status}`, res.status);
    }
    return schema.parse(data);
  }

  const s = (id: string) => `/sessions/${encodeURIComponent(id)}`;
  return {
    openSession: (p: OpenSessionParams) =>
      call(SessionView, '/sessions', { method: 'POST', body: p }),
    getSession: (id: string) => call(SessionView, s(id)),
    candidates: (id: string, item: number, k?: number) =>
      k === undefined
        ? call(CandidateView.array(), `${s(id)}/candidates/${item}`)
        : call(CandidateView.array(), `${s(id)}/candidates/${item}`, {
            method: 'POST',
            body: { k },
          }),
    edit: (id: string, edits: Edit[]) =>
      call(EditResult, `${s(id)}/edits`, { method: 'POST', body: { edits } }),
    label: (id: string, item: number, candidate: number, isMatch: boolean) =>
      call(LabelResult, `${s(id)}/label`, {
        method: 'POST',
        body: { item, candidate, is_match: isMatch },
      }),
    next: (id: string) => call(Query.nullable(), `${s(id)}/next`),
  };
}

export type Client = ReturnType<typeof createClient>;
