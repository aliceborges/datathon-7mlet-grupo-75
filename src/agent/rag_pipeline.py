"""Pipeline RAG com Chroma: indexa o catálogo de ofertas e o glossário de features."""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "rag_corpus"
DEFAULT_COLLECTION = "datathon_rag"

_H2_SPLIT = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def split_markdown_by_h2(text: str) -> list[tuple[str, str]]:
    """Quebra um markdown em pares (título, corpo) usando ## como separador."""
    sections: list[tuple[str, str]] = []
    matches = list(_H2_SPLIT.finditer(text))
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


@dataclass
class RAGPipeline:
    """Wrapper fino sobre uma collection Chroma."""

    collection: Collection

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        result = self.collection.query(query_texts=[query], n_results=k)
        docs = result.get("documents") or [[]]
        return docs[0] if docs else []

    def ingest_markdown(self, path: Path, source_tag: str | None = None) -> int:
        sections = split_markdown_by_h2(path.read_text(encoding="utf-8"))
        if not sections:
            logger.warning("Nenhuma seção ## encontrada em %s", path.name)
            return 0
        tag = source_tag or path.stem
        ids = [f"{tag}:{title}" for title, _ in sections]
        docs = [f"[{title}] {body}" for title, body in sections]
        metas = [{"source": tag, "title": title} for title, _ in sections]
        self.collection.upsert(documents=docs, metadatas=metas, ids=ids)  # type: ignore[arg-type]
        return len(sections)


def build_default_pipeline(
    corpus_dir: Path | None = None,
    client: ClientAPI | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> RAGPipeline:
    """Cria pipeline em memória e ingere todos os .md de data/rag_corpus/."""
    chroma_client = client or chromadb.EphemeralClient()
    with contextlib.suppress(Exception):
        chroma_client.delete_collection(name=collection_name)
    collection = chroma_client.create_collection(name=collection_name)
    pipeline = RAGPipeline(collection=collection)

    root = corpus_dir or DEFAULT_CORPUS_DIR
    for md_file in sorted(root.glob("*.md")):
        ingested = pipeline.ingest_markdown(md_file)
        logger.info("RAG ingestion: %s (%d seções)", md_file.name, ingested)

    return pipeline
