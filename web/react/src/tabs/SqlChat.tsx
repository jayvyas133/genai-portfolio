import { type FormEvent, useRef, useState } from "react";
import { type Cell, type Turn, errorText, postJson, seconds } from "../api";

type NewMessage =
  | { role: "user"; text: string }
  | { role: "bot"; turn: Turn }
  | { role: "error"; text: string };

type Message = NewMessage & { id: number };

export function SqlChat() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [waiting, setWaiting] = useState(false);
  const conversationId = useRef<string | null>(null);
  const nextId = useRef(0);

  const push = (m: NewMessage) => setMessages((all) => [...all, { ...m, id: nextId.current++ }]);

  function newConversation() {
    conversationId.current = null;
    setMessages([]);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    setQuestion("");
    push({ role: "user", text });
    // Disabled while a turn runs: the API allows one turn at a time per conversation (409).
    setWaiting(true);
    try {
      if (conversationId.current === null) {
        conversationId.current = (await postJson<{ id: string }>("/api/sql/conversations")).id;
      }
      const turn = await postJson<Turn>(`/api/sql/conversations/${conversationId.current}/messages`, {
        question: text,
      });
      push({ role: "bot", turn });
    } catch (err) {
      push({ role: "error", text: errorText(err) });
    } finally {
      setWaiting(false);
    }
  }

  return (
    <>
      <p className="lede">
        Chat with a music-store database (Chinook). Follow-ups such as "Only for 2024." build on the
        previous question. Queries are read-only.
      </p>
      <div className="chat" aria-live="polite">
        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="msg user">
              {m.text}
            </div>
          ) : m.role === "error" ? (
            <div key={m.id} className="msg bot">
              <p className="error">{m.text}</p>
            </div>
          ) : (
            <BotReply key={m.id} turn={m.turn} />
          ),
        )}
        {waiting && (
          <div className="msg bot">
            <p className="working">Thinking...</p>
          </div>
        )}
      </div>
      <form onSubmit={submit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. What is the total revenue for each billing country?"
          maxLength={2000}
          required
          disabled={waiting}
        />
        <button type="submit" disabled={waiting}>
          Send
        </button>
        <button type="button" className="secondary" onClick={newConversation} disabled={waiting}>
          New conversation
        </button>
      </form>
    </>
  );
}

function BotReply({ turn }: { turn: Turn }) {
  const rows = `${turn.rows.length}${turn.truncated ? "+" : ""}`;
  return (
    <div className="msg bot">
      <p>{turn.answer ?? "(no answer)"}</p>
      <details>
        <summary>{turn.sql ? `SQL and ${rows} row(s)` : "Details"}</summary>
        {turn.sql && <pre>{turn.sql}</pre>}
        {turn.columns.length > 0 && <ResultTable columns={turn.columns} rows={turn.rows} />}
        {turn.error && (
          <p className="error">
            {turn.error_kind}: {turn.error}
          </p>
        )}
      </details>
      <p className="meta">
        {seconds(turn.latency_ms)}
        {turn.attempts > 1 && `, repaired after ${turn.attempts} attempts`}
      </p>
    </div>
  );
}

function ResultTable({ columns, rows }: { columns: string[]; rows: Cell[][] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 50).map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j}>{v === null ? "NULL" : String(v)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
