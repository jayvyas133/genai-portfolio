import { useState } from "react";
import { DocuMind } from "./tabs/DocuMind";
import { Research } from "./tabs/Research";
import { SqlChat } from "./tabs/SqlChat";

const TABS = [
  { id: "documind", label: "DocuMind" },
  { id: "research", label: "Deep Research" },
  { id: "sql", label: "Text-to-SQL chat" },
] as const;

type TabId = (typeof TABS)[number]["id"];

function initialTab(): TabId {
  const hash = window.location.hash.slice(1);
  return TABS.some((t) => t.id === hash) ? (hash as TabId) : "documind";
}

export function App() {
  const [tab, setTab] = useState<TabId>(initialTab);

  const select = (id: TabId) => {
    setTab(id);
    window.history.replaceState(null, "", `#${id}`);
  };

  return (
    <>
      <header>
        <h1>GenAI Portfolio</h1>
        <p>
          Three AI (Artificial Intelligence) services behind one URL. React version; the{" "}
          <a href="/">plain page</a> does the same without a framework.
        </p>
        <nav role="tablist">
          {TABS.map((t) => (
            <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => select(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {/* Kept mounted (hidden, not removed) so switching tabs keeps each tab's state. */}
        <section className="panel" hidden={tab !== "documind"}>
          <DocuMind />
        </section>
        <section className="panel" hidden={tab !== "research"}>
          <Research />
        </section>
        <section className="panel" hidden={tab !== "sql"}>
          <SqlChat />
        </section>
      </main>
    </>
  );
}
