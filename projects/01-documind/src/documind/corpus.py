"""Load the source documents that DocuMind answers questions about.

The corpus is a folder of JSON files, one per document, each holding
``title``, ``url`` and ``text``. Text marks sections with
``== Heading ==`` lines, which the chunker uses to split it.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """One source document.

    ``title`` identifies the document in chunks and answers.
    ``url`` is where it came from; ``text`` is the full plain text.
    """

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    url: str
    text: str = Field(min_length=1)


def load_documents(folder: Path) -> list[Document]:
    """Load every ``*.json`` document in ``folder``, sorted by file name.

    Sorting keeps the order stable across machines, so ingestion
    and evals are reproducible.

    Raises:
        FileNotFoundError: If ``folder`` does not exist.
        ValueError: If the folder holds no documents.
        pydantic.ValidationError: If a file does not match ``Document``.
    """
    if not folder.is_dir():
        msg = f"Corpus folder not found: {folder}"
        raise FileNotFoundError(msg)
    paths = sorted(folder.glob("*.json"))
    if not paths:
        msg = f"No documents (*.json) in {folder}"
        raise ValueError(msg)
    return [Document.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]
