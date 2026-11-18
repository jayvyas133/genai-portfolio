"""The research corpus: a frozen snapshot of Wikipedia articles.

Research runs against this snapshot, not the live web, so the same
question always sees the same sources. That makes eval runs reproducible,
free and offline (ADR 0015). ``fetch_documents`` builds the snapshot once;
the JSON files are committed and ``load_documents`` reads them.

Text marks sections with ``== Heading ==`` lines (Wikipedia's own
format), which the passage splitter uses.
"""

import asyncio
import re
from collections.abc import Iterable
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

# Topic: space exploration and launch vehicles. Chosen because it is full of
# questions that need several articles at once (comparisons between
# agencies, rockets and missions; timelines across programmes), which is
# exactly what a single search-and-answer step handles badly.
TITLES: tuple[str, ...] = (
    "ISRO",
    "Polar Satellite Launch Vehicle",
    "Geosynchronous Satellite Launch Vehicle",
    "LVM3",
    "Chandrayaan-1",
    "Chandrayaan-2",
    "Chandrayaan-3",
    "Mars Orbiter Mission",
    "Gaganyaan",
    "NASA",
    "Apollo program",
    "Apollo 11",
    "Saturn V",
    "Space Shuttle",
    "Artemis program",
    "Space Launch System",
    "Orion (spacecraft)",
    "International Space Station",
    "Hubble Space Telescope",
    "James Webb Space Telescope",
    "Curiosity (rover)",
    "Perseverance (rover)",
    "SpaceX",
    "Falcon 9",
    "Falcon Heavy",
    "SpaceX Starship",
    "Reusable launch vehicle",
    "Blue Origin",
    "New Glenn",
    "China National Space Administration",
    "Tiangong space station",
    "Chang'e 4",
    "Chang'e 5",
    "Luna 25",
    "Roscosmos",
    "European Space Agency",
    "Ariane 5",
    "Ariane 6",
)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Generic client name: Wikipedia asks automated clients to identify
# themselves, but no personal contact details go in it.
USER_AGENT = "genai-portfolio-deep-research/0.1 (corpus snapshot script)"

# Trailing sections that hold links and references, not facts.
_DROP_SECTIONS = frozenset(
    {
        "see also",
        "references",
        "notes",
        "external links",
        "further reading",
        "bibliography",
        "citations",
        "sources",
        "footnotes",
        "general and cited references",
    }
)

_HEADING_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$", re.MULTILINE)


class Document(BaseModel):
    """One source document: ``title`` identifies it, ``url`` is its origin."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    url: str
    text: str = Field(min_length=1)


def slugify(title: str) -> str:
    """File-name-safe form of ``title``: ``"Chang'e 4"`` -> ``"change-4"``."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower().replace("'", ""))
    return slug.strip("-")


def strip_trailing_sections(text: str) -> str:
    """Drop reference-style sections ("See also", "References", ...) and their bodies.

    A dropped section's subsections go with it; the next heading at the
    same or a higher level ends the drop.
    """
    kept: list[str] = []
    drop_level: int | None = None
    pos = 0
    for match in _HEADING_RE.finditer(text):
        if drop_level is None:
            kept.append(text[pos : match.start()])
        level = len(match.group(1))
        if drop_level is not None and level <= drop_level:
            drop_level = None
        if drop_level is None and match.group(2).strip().lower() in _DROP_SECTIONS:
            drop_level = level
        if drop_level is None:
            kept.append(match.group(0))
        pos = match.end()
    if drop_level is None:
        kept.append(text[pos:])
    joined = "".join(kept)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


async def fetch_document(
    client: httpx.AsyncClient, title: str, *, max_retries: int = 3
) -> Document:
    """Download one article's plain text from the Wikipedia API.

    A 429 (Too Many Requests) reply is retried after the server's
    ``Retry-After`` delay (5 s if it sends none), up to ``max_retries`` times.

    Raises:
        httpx.HTTPError: If the request fails or is still rate-limited.
        LookupError: If the page does not exist.
    """
    params = {
        "action": "query",
        "prop": "extracts|info",
        "inprop": "url",
        "explaintext": "1",
        "exsectionformat": "wiki",
        "redirects": "1",
        "titles": title,
        "format": "json",
        "formatversion": "2",
    }
    for attempt in range(max_retries + 1):
        response = await client.get(WIKIPEDIA_API, params=params)
        if response.status_code != 429 or attempt == max_retries:
            break
        await asyncio.sleep(float(response.headers.get("Retry-After", "5")))
    response.raise_for_status()
    page = response.json()["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        raise LookupError(f"Wikipedia page not found: {title}")
    return Document(
        title=page["title"],
        url=page["fullurl"],
        text=strip_trailing_sections(page["extract"]),
    )


async def fetch_documents(
    titles: Iterable[str], folder: Path, *, delay_s: float = 2.0
) -> list[Path]:
    """Fetch each title not already in ``folder`` and save it as JSON.

    Requests go one at a time with ``delay_s`` between them, to stay
    under Wikipedia's rate limit. Existing files are kept, so an
    interrupted run resumes where it stopped; delete a file to refresh it.
    """
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        for title in titles:
            path = folder / f"{slugify(title)}.json"
            if not path.exists():
                doc = await fetch_document(client, title)
                path.write_text(doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
                await asyncio.sleep(delay_s)
            paths.append(path)
    return paths


def load_documents(folder: Path) -> list[Document]:
    """Load every ``*.json`` document in ``folder``, sorted by file name.

    Raises:
        FileNotFoundError: If ``folder`` does not exist.
        ValueError: If the folder holds no documents.
        pydantic.ValidationError: If a file does not match ``Document``.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Corpus folder not found: {folder}")
    paths = sorted(folder.glob("*.json"))
    if not paths:
        raise ValueError(f"No documents (*.json) in {folder}")
    return [Document.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]
