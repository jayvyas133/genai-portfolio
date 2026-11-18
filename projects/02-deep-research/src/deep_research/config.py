"""Configuration for Deep Research.

Values come from environment variables prefixed with ``DEEP_RESEARCH_``
(or a local ``.env`` file). Model choice and API keys are not here: they
belong to aicore (``AICORE_*``), which makes every model call.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from deep_research.agents import AgentLimits

# projects/02-deep-research/ (this file is src/deep_research/config.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DeepResearchSettings(BaseSettings):
    """Settings for search, agents, evals and the research service.

    Example: ``DEEP_RESEARCH_SEARCH_K=8`` sets ``search_k`` to 8.
    """

    model_config = SettingsConfigDict(
        env_prefix="DEEP_RESEARCH_",
        env_file=".env",
        extra="ignore",
    )

    # Frozen snapshot of source documents, one JSON file each.
    corpus_dir: Path = _PROJECT_ROOT / "data" / "corpus"
    # Golden dataset (golden.jsonl) and saved eval results (results/).
    evals_dir: Path = _PROJECT_ROOT / "evals"
    # Passages returned per search query.
    search_k: int = Field(default=5, gt=0, le=50)
    # Target passage length. ~120 words is one or two paragraphs: long
    # enough to hold a fact with its context, short enough that five of
    # them fit a small local model's context window with room to spare.
    passage_words: int = Field(default=120, gt=20)
    # Agent budgets (see AgentLimits): they cap LLM calls per report.
    max_sub_questions: int = Field(default=3, ge=1, le=6)
    queries_per_sub_question: int = Field(default=2, ge=1, le=5)
    passages_per_sub_question: int = Field(default=4, ge=1, le=10)
    max_revisions: int = Field(default=1, ge=0, le=3)

    # Research service (API + worker), see jobs.py and worker.py.
    redis_url: str = "redis://localhost:6379/0"
    # Jobs one worker process runs at once. 1 suits a local Ollama, which
    # serves one request at a time anyway; raise it for a hosted model.
    worker_concurrency: int = Field(default=1, ge=1, le=16)
    # Queued + running jobs; past it the API answers 429 (back-pressure)
    # instead of letting the queue grow without bound.
    max_pending_jobs: int = Field(default=20, ge=1)
    # Hard cap on one job's run time. A 3B local model takes ~4 minutes
    # per report (ADR 0016), so 15 minutes only stops runaway jobs.
    job_timeout_s: float = Field(default=900.0, gt=0)
    # How long finished jobs (and their reports and events) are kept.
    result_ttl_s: int = Field(default=86_400, gt=0)

    def agent_limits(self) -> AgentLimits:
        """The agent budgets as one object."""
        return AgentLimits(
            max_sub_questions=self.max_sub_questions,
            queries_per_sub_question=self.queries_per_sub_question,
            passages_per_sub_question=self.passages_per_sub_question,
            max_revisions=self.max_revisions,
        )
