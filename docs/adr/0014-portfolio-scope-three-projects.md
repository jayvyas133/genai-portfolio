# ADR (Architecture Decision Record) 0014: Portfolio scope, three projects

- **Status:** accepted
- **Date:** 2026-11-18
- **Scope:** portfolio

## Context
The repository was scaffolded with five projects: DocuMind, a Text-to-SQL
(Structured Query Language) agent, an SLM (Small Language Model)
fine-tune, Deep Research and an Indic voice agent. With DocuMind done, the
remaining time is better spent on fewer, deeper projects. One of them must
be a multi-agent system, and one should be a chatbot, meaning multi-turn
conversation with memory.

## Options considered
1. **Keep all five.** Widest coverage, but the fine-tune needs paid GPU
   (Graphics Processing Unit) time and the voice agent needs audio models;
   both are expensive and hard for a reviewer to run.
2. **A separate simple chatbot project.** Shows conversation, but on its own
   it reads as a tutorial and adds little next to the other projects.
3. **Three projects that build on each other** (chosen): retrieval, then
   several agents coordinating, then a conversational agent.

## Decision
- **01 DocuMind** (done): RAG (Retrieval-Augmented Generation), Flask.
- **02 Deep Research**: the multi-agent project. A planner, parallel
  researchers, a writer and a critic, orchestrated with LangGraph and served
  by FastAPI. Research is a natural multi-agent problem because the
  sub-questions are independent and can run in parallel.
- **03 Text-to-SQL Agent**: the only chatbot. Multi-turn memory is
  required here, not decoration, because follow-ups such as "break that
  down by region" only make sense against the previous query. Flask +
  LangGraph.
- The SLM fine-tune and the Indic voice agent are dropped.
- LangGraph is allowed wherever an agent graph fits (02 and 03), not only
  in one project. LangChain stays out.

## Trade-offs
- **No model training or voice work** in the portfolio. Those skills are
  not shown; that is accepted to keep everything runnable on a free tier.
- **Two web frameworks.** FastAPI in 02 suits long-running, streamed
  research jobs; Flask stays in 01 and 03. The cost is two stacks to
  maintain; the benefit is showing both.
- **Earlier ADRs say "five projects".** They are left as written, since an
  ADR records what was true when it was accepted.
