import { type FormEvent, useState } from "react";
import { type AskResponse, errorText, postJson, seconds } from "../api";

type State =
  | { kind: "idle" }
  | { kind: "working" }
  | { kind: "answer"; data: AskResponse }
  | { kind: "error"; message: string };

export function DocuMind() {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });

  async function submit(event: FormEvent) {
    event.preventDefault();
    setState({ kind: "working" });
    try {
      const data = await postJson<AskResponse>("/api/documind/ask", { question: question.trim() });
      setState({ kind: "answer", data });
    } catch (err) {
      setState({ kind: "error", message: errorText(err) });
    }
  }

  const working = state.kind === "working";
  return (
    <>
      <p className="lede">
        Question answering over a document collection with cited sources. RAG (Retrieval-Augmented
        Generation): hybrid search, reranking, refusals.
      </p>
      <form onSubmit={submit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about the documents..."
          maxLength={1000}
          required
          disabled={working}
        />
        <button type="submit" disabled={working}>
          Ask
        </button>
      </form>
      {state.kind !== "idle" && (
        <div className="out">
          {state.kind === "working" && <p className="working">Searching the documents...</p>}
          {state.kind === "error" && <p className="error">{state.message}</p>}
          {state.kind === "answer" && <Answer data={state.data} />}
        </div>
      )}
    </>
  );
}

function Answer({ data }: { data: AskResponse }) {
  return (
    <>
      <p>{data.refused ? data.message : data.answer}</p>
      {data.citations.length > 0 && (
        <ol className="sources">
          {data.citations.map((c) => (
            <li key={c.number} value={c.number}>
              {c.document_title} - {c.section}
            </li>
          ))}
        </ol>
      )}
      <p className="meta">
        {data.refused ? `Refused (${data.refusal_reason})` : "Answered"} in {seconds(data.latency_ms)}
      </p>
    </>
  );
}
