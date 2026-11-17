"""Retrieval metrics: did the retriever find the passages that answer a question?

For each golden case, the retriever returns a ranked list of chunks.
A chunk is relevant to an evidence passage when it comes from the same
document and contains the quote (case and whitespace ignored), or matches
one of the passage's alternative sources.

Per case we record only the rank (1-based) at which each evidence passage
was first found, or ``None`` if it was not retrieved. Every metric is then
a function of those ranks, for any cut-off ``k``:

- **hit rate@k**: share of cases with at least one evidence passage in
  the top k. "Did the LLM see anything useful?"
- **recall@k**: share of a case's evidence passages found in the top k,
  averaged over cases. Differs from hit rate only for multi-hop cases,
  where the answer needs every passage.
- **MRR** (Mean Reciprocal Rank): average of 1 / rank of the first
  relevant chunk (0 if none). Rewards putting the answer first, which
  matters because the LLM reads the top chunks most reliably.

Unanswerable cases have no evidence, so they are left out of retrieval
metrics; they are scored on refusal in M5 (generation evals).
"""

import math
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, computed_field

from documind.golden import Evidence, GoldenCase, Kind, normalize


class RankedChunk(Protocol):
    """Anything with a document title and chunk text, such as ``Chunk``."""

    @property
    def document_title(self) -> str:
        """Title of the document the chunk came from."""
        ...

    @property
    def text(self) -> str:
        """The chunk's text, without the embedding header."""
        ...


def is_relevant(evidence: Evidence, chunk: RankedChunk) -> bool:
    """True when ``chunk`` matches any of the evidence's sources.

    A source matches when the chunk is from its document and contains its quote.
    """
    text = normalize(chunk.text)
    return any(
        chunk.document_title == source.title and normalize(source.quote) in text
        for source in evidence.sources
    )


class CaseRetrieval(BaseModel):
    """Where one case's evidence appeared in the ranked results.

    ``evidence_ranks[i]`` is the 1-based rank of the first chunk matching
    evidence ``i``, or ``None`` if no retrieved chunk matched it.
    ``retrieved`` is how many chunks the retriever returned (the depth).
    ``latency_ms`` is the retrieval time, when the caller measured it.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str
    kind: Kind
    retrieved: int = Field(ge=0)
    evidence_ranks: list[int | None]
    latency_ms: float | None = Field(default=None, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def first_rank(self) -> int | None:
        """Rank of the first relevant chunk for any evidence, or ``None``."""
        found = [r for r in self.evidence_ranks if r is not None]
        return min(found) if found else None

    def hit(self, k: int) -> bool:
        """True when some evidence passage is in the top ``k``."""
        return self.first_rank is not None and self.first_rank <= k

    def recall(self, k: int) -> float:
        """Share of evidence passages found in the top ``k``."""
        found = sum(1 for r in self.evidence_ranks if r is not None and r <= k)
        return found / len(self.evidence_ranks)

    def reciprocal_rank(self) -> float:
        """1 / rank of the first relevant chunk, or 0.0 if none was retrieved."""
        return 0.0 if self.first_rank is None else 1.0 / self.first_rank


def score_case(
    case: GoldenCase, ranked: Sequence[RankedChunk], latency_ms: float | None = None
) -> CaseRetrieval:
    """Record the rank of each of ``case``'s evidence passages in ``ranked``.

    Raises:
        ValueError: If ``case`` is unanswerable (it has no evidence to find).
    """
    if not case.evidence:
        msg = f"{case.id}: unanswerable cases have no retrieval score"
        raise ValueError(msg)
    ranks = [
        next((i for i, chunk in enumerate(ranked, start=1) if is_relevant(ev, chunk)), None)
        for ev in case.evidence
    ]
    return CaseRetrieval(
        case_id=case.id,
        kind=case.kind,
        retrieved=len(ranked),
        evidence_ranks=ranks,
        latency_ms=latency_ms,
    )


class RetrievalReport(BaseModel):
    """Retrieval metrics over a set of scored cases.

    Metrics are means over ``cases``; ``hits(k)`` gives the raw count,
    because with 40 cases one case moves hit rate by 2.5 points.
    """

    model_config = ConfigDict(frozen=True)

    cases: list[CaseRetrieval] = Field(min_length=1)

    def hits(self, k: int) -> int:
        """Number of cases with some evidence in the top ``k``."""
        return sum(c.hit(k) for c in self.cases)

    def hit_rate(self, k: int) -> float:
        """Share of cases with some evidence in the top ``k``."""
        return self.hits(k) / len(self.cases)

    def recall(self, k: int) -> float:
        """Mean recall@k over cases."""
        return sum(c.recall(k) for c in self.cases) / len(self.cases)

    def mrr(self) -> float:
        """Mean reciprocal rank over the full retrieved depth."""
        return sum(c.reciprocal_rank() for c in self.cases) / len(self.cases)

    def only(self, kind: Kind) -> "RetrievalReport | None":
        """The same report restricted to ``kind``, or ``None`` if it has no cases."""
        cases = [c for c in self.cases if c.kind == kind]
        return RetrievalReport(cases=cases) if cases else None

    def latency_percentile(self, pct: float) -> float | None:
        """Nearest-rank percentile of measured latencies in ms, or ``None`` if none.

        Raises:
            ValueError: If ``pct`` is not in (0, 100].
        """
        if not 0 < pct <= 100:
            msg = f"pct must be in (0, 100], got {pct}"
            raise ValueError(msg)
        values = sorted(c.latency_ms for c in self.cases if c.latency_ms is not None)
        if not values:
            return None
        return values[max(math.ceil(pct / 100 * len(values)) - 1, 0)]
