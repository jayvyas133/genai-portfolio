// Types and fetch helpers for the three APIs behind the gateway.
// The shapes mirror the Pydantic response models in each project.

export type Citation = { number: number; document_title: string; section: string };

export type AskResponse = {
  answer: string;
  refused: boolean;
  refusal_reason: string | null;
  message: string | null;
  citations: Citation[];
  latency_ms: number;
};

export type Source = { id: string; title: string; section: string; url: string };

export type Job = {
  id: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  error: string | null;
  report: { text: string; sources: Source[] } | null;
};

export type ResearchEvent = { id: string; type: string; message: string; node: string | null };

export type Cell = string | number | boolean | null;

export type Turn = {
  conversation_id: string;
  question: string;
  answer: string | null;
  sql: string | null;
  columns: string[];
  rows: Cell[][];
  truncated: boolean;
  attempts: number;
  error_kind: string | null;
  error: string | null;
  latency_ms: number;
};

/** Parse a response; throws an Error carrying the API's ``detail`` on failure. */
export async function readJson<T>(response: Response): Promise<T> {
  let data: unknown = null;
  try {
    data = await response.json();
  } catch {
    // Not JSON, e.g. a 502 page from nginx.
  }
  if (!response.ok) {
    const body = data as { detail?: unknown; error?: unknown } | null;
    const detail = body?.detail ?? body?.error;
    throw new Error(`${response.status}: ${typeof detail === "string" ? detail : response.statusText}`);
  }
  return data as T;
}

export async function postJson<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return readJson<T>(response);
}

export const seconds = (ms: number): string => `${(ms / 1000).toFixed(1)} s`;

export const errorText = (err: unknown): string => (err instanceof Error ? err.message : String(err));
