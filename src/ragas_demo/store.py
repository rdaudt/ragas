from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from openai import OpenAI

from ragas_demo.models import DocumentChunk, SourceChunk


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class VectorIndex:
    def __init__(
        self,
        persist_path: Path,
        embedder: Embedder,
        collection_name: str = "ragas-demo",
    ) -> None:
        persist_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.embedder = embedder
        self.collection_name = collection_name

    def _collection(self) -> Collection:
        return self.client.get_collection(self.collection_name)

    def matches(self, corpus_fingerprint: str) -> bool:
        try:
            collection = self._collection()
        except NotFoundError:
            return False
        return collection.metadata.get("corpus_fingerprint") == corpus_fingerprint

    def replace(
        self,
        chunks: Sequence[DocumentChunk],
        corpus_fingerprint: str,
        batch_size: int = 100,
    ) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except NotFoundError:
            pass
        collection = self.client.create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine", "corpus_fingerprint": corpus_fingerprint},
        )
        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start : start + batch_size])
            documents = [chunk.text for chunk in batch]
            collection.add(
                ids=[chunk.id for chunk in batch],
                documents=documents,
                embeddings=self.embedder.embed(documents),
                metadatas=[
                    {
                        "source_file": chunk.source_file,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                    }
                    for chunk in batch
                ],
            )

    def query(self, question: str, top_k: int) -> list[SourceChunk]:
        collection = self._collection()
        result_count = min(top_k, collection.count())
        if result_count == 0:
            return []
        result = collection.query(
            query_embeddings=self.embedder.embed([question]),
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []
        return [
            SourceChunk(
                id=identifier,
                text=document,
                source_file=str(metadata["source_file"]),
                page=int(metadata["page"]),
                chunk_index=int(metadata["chunk_index"]),
                score=max(-1.0, min(1.0, 1.0 - float(distance))),
            )
            for identifier, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]
