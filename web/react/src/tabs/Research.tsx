import { type FormEvent, useEffect, useRef, useState } from "react";
import { type Job, type ResearchEvent, errorText, postJson, readJson, seconds } from "../api";

const FINAL = ["done", "failed", "cancelled"] as const;

type Step = { text: string; final: boolean };

export function Research() {
  const [question, setQuestion] = useState("");
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState<Step[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const source = useRef<EventSource | null>(null);

  // Close the stream if the component goes away mid-job.
  useEffect(() => () => source.current?.close(), []);

  const add = (text: string, final = false) => setSteps((s) => [...s, { text, final }]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSteps([]);
    setJob(null);
    setError(null);
    setRunning(true);
    try {
      const queued = await postJson<Job>("/api/research/research", { question: question.trim() });
      follow(queued.id);
    } catch (err) {
      setError(errorText(err));
      setRunning(false);
    }
  }

  // The API replays past events first, then streams new ones until a final one.
  function follow(id: string) {
    const started = performance.now();
    const events = new EventSource(`/api/research/research/${id}/events`);
    source.current = events;
    events.addEventListener("queued", () => add("Queued"));
    events.addEventListener("started", () => add("Started"));
    events.addEventListener("progress", (e) => {
      const data = JSON.parse((e as MessageEvent<string>).data) as ResearchEvent;
      add(data.node ? `[${data.node}] ${data.message}` : data.message);
    });
    for (const type of FINAL) {
      events.addEventListener(type, async (e) => {
        events.close(); // final event: stop EventSource from reconnecting
        add(`${type} after ${seconds(performance.now() - started)}`, true);
        try {
          if (type === "done") setJob(await readJson<Job>(await fetch(`/api/research/research/${id}`)));
          else setError((JSON.parse((e as MessageEvent<string>).data) as ResearchEvent).message);
        } catch (err) {
          setError(errorText(err));
        }
        setRunning(false);
      });
    }
  }

  const report = job?.report;
  return (
    <>
      <p className="lede">
        A planner, parallel researchers, a writer and a critic produce a cited report. Progress streams
        live; a report takes a few minutes on a local model.
      </p>
      <form onSubmit={submit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. Compare Chandrayaan-3 and Luna 25"
          minLength={3}
          maxLength={500}
          required
          disabled={running}
        />
        <button type="submit" disabled={running}>
          Research
        </button>
      </form>
      {steps.length > 0 && (
        <ol className="progress">
          {steps.map((s, i) => (
            <li key={i} className={s.final ? "final" : ""}>
              {s.text}
            </li>
          ))}
        </ol>
      )}
      {(report || error) && (
        <div className="out">
          {error && <p className="error">{error}</p>}
          {report && (
            <>
              {report.text.split(/\n{2,}/).map((p, i) => (
                <p key={i}>{p}</p>
              ))}
              <ol className="sources">
                {report.sources.map((s) => (
                  <li key={s.id}>
                    <a href={s.url} target="_blank" rel="noopener">
                      {s.title} - {s.section}
                    </a>
                  </li>
                ))}
              </ol>
            </>
          )}
        </div>
      )}
    </>
  );
}
