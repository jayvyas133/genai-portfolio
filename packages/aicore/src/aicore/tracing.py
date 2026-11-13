"""Tracing for LLM calls.

``LLMClient`` emits one ``CallRecord`` per
logical call (all retries included) to a
``Tracer``. Records are plain Pydantic data, so
any backend (logs, Langfuse) can consume them.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from aicore.models import Usage


def _new_call_id() -> str:
    """Return a random, unique ID for one call."""
    return uuid.uuid4().hex


class CallRecord(BaseModel):
    """What happened in one ``LLMClient`` call.

    ``usage`` and ``cost_usd`` are ``None`` when
    the call failed, or when the cost is unknown.
    ``outcome`` is ``"ok"`` or the error's class
    name, for example ``"RateLimitError"``.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(default_factory=_new_call_id)
    provider: str
    model: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latency_ms: float = Field(ge=0)
    attempts: int = Field(ge=1)
    outcome: str
    usage: Usage | None = None
    cost_usd: float | None = Field(default=None, ge=0)


class Tracer(Protocol):
    """Anything that can receive call records."""

    def record(self, rec: CallRecord) -> None:
        """Store or export one finished call."""
        ...


class NullTracer:
    """Tracer that drops every record.

    The client's default, so tracing costs
    nothing unless a project opts in.
    """

    def record(self, rec: CallRecord) -> None:
        """Ignore ``rec``."""


class InMemoryTracer:
    """Tracer that keeps records in a list.

    Meant for tests and quick local checks.
    """

    def __init__(self) -> None:
        """Start with no records."""
        self.records: list[CallRecord] = []

    def record(self, rec: CallRecord) -> None:
        """Append ``rec`` to ``self.records``."""
        self.records.append(rec)


class LogTracer:
    """Tracer that logs each record as JSON.

    One INFO line per call on the
    ``aicore.trace`` logger, so it works with
    any log shipper without new dependencies.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """Use ``logger``, or ``aicore.trace``."""
        self._log = logger or logging.getLogger("aicore.trace")

    def record(self, rec: CallRecord) -> None:
        """Log ``rec`` as a single JSON line."""
        self._log.info(rec.model_dump_json())
