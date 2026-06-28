from __future__ import annotations

import uuid

import chromadb
import pytest

from src.agent.rag_pipeline import (
    RAGPipeline,
    split_markdown_by_h2,
)


class _FixedEmbedding:
    """Embedding determinístico para testes — evita download do modelo padrão."""

    def __call__(self, input):  # noqa: A002
        return [
            [float(ord(c) % 7) for c in (text[:8] or " ").ljust(8)] for text in input
        ]

    def embed_query(self, input):  # noqa: A002
        texts = input if isinstance(input, list) else [input]
        return self(texts)

    def embed_documents(self, input):  # noqa: A002
        return self(input)

    def name(self) -> str:
        return "fixed-test"


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    name = f"test_rag_{uuid.uuid4().hex[:8]}"
    return client.create_collection(name=name, embedding_function=_FixedEmbedding())


class TestSplitMarkdownByH2:
    def test_extracts_sections(self):
        text = "intro\n\n## A\n\nbody A\n\n## B\n\nbody B"
        result = split_markdown_by_h2(text)
        assert result == [("A", "body A"), ("B", "body B")]

    def test_skips_empty_sections(self):
        text = "## A\n\nbody A\n\n## empty\n\n## B\n\nbody B"
        result = split_markdown_by_h2(text)
        titles = [t for t, _ in result]
        assert "empty" not in titles
        assert titles == ["A", "B"]

    def test_returns_empty_when_no_h2(self):
        assert split_markdown_by_h2("texto sem headers") == []


class TestRAGPipeline:
    def test_ingest_returns_section_count(self, collection, tmp_path):
        md = tmp_path / "sample.md"
        md.write_text("## one\n\ndoc one\n\n## two\n\ndoc two", encoding="utf-8")
        pipeline = RAGPipeline(collection=collection)
        assert pipeline.ingest_markdown(md) == 2

    def test_retrieve_returns_documents(self, collection, tmp_path):
        md = tmp_path / "sample.md"
        md.write_text("## one\n\ndoc one\n\n## two\n\ndoc two", encoding="utf-8")
        pipeline = RAGPipeline(collection=collection)
        pipeline.ingest_markdown(md)
        docs = pipeline.retrieve("query", k=2)
        assert len(docs) == 2

    def test_ingest_uses_filename_as_source_tag(self, collection, tmp_path):
        md = tmp_path / "offers.md"
        md.write_text("## a\n\nbody", encoding="utf-8")
        pipeline = RAGPipeline(collection=collection)
        pipeline.ingest_markdown(md)
        # ID gerado deve usar o stem
        assert collection.get(ids=["offers:a"])["ids"] == ["offers:a"]
