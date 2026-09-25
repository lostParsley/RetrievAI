"""End-to-end RAG pipeline for loading, indexing, retrieving, and answering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from uuid import uuid4

from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from app.config import Settings


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class SentenceTransformerEmbeddings(Embeddings):
    """Tiny adapter so LangChain can call sentence-transformers directly.

    LangChain expects an object with ``embed_documents`` and ``embed_query``.
    We keep the real embedding model as sentence-transformers and use this
    adapter only so LangChain's Qdrant retrieval wrapper can call it.
    """

    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


class RAGPipeline:
    """A tiny, readable RAG pipeline that makes each step explicit.

    Responsibilities:
    - load local documents
    - split documents into chunks
    - embed chunks
    - create and populate the Qdrant collection
    - retrieve the top-k nearest chunks for a question
    - build a grounded prompt
    - generate the final answer
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Qdrant is the persistent vector store. The client talks to the
        # Qdrant server running in Docker via HTTP.
        self.qdrant = QdrantClient(url=settings.qdrant_url)

        # sentence-transformers creates embeddings for both document chunks
        # and user questions.
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.langchain_embeddings = SentenceTransformerEmbeddings(self.embedder)

        # A small local generation model is used only after retrieval.
        # It receives retrieved context and produces the final grounded answer.
        self.answer_generator = pipeline(
            "text2text-generation",
            model=settings.answer_model,
            tokenizer=settings.answer_model,
            max_new_tokens=180,
        )

    def index_documents(self, docs_path: str | Path | None = None) -> dict:
        """Run the indexing pipeline from local files into Qdrant."""

        folder = Path(docs_path or self.settings.docs_path).resolve()
        documents = self._load_documents(folder)
        chunks = self._split_documents(documents)
        points = self._build_points(chunks)

        # For teaching clarity, each indexing run rebuilds the collection from
        # scratch so you can reason about exactly what is stored.
        self._recreate_collection()
        if points:
            self.qdrant.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=points,
            )

        return {
            "collection_name": self.settings.qdrant_collection_name,
            "documents_loaded": len(documents),
            "chunks_indexed": len(points),
            "embedding_model": self.settings.embedding_model,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "indexed_chunks": [
                {
                    "id": point.id,
                    "source": point.payload["metadata"]["source"],
                    "chunk_index": point.payload["metadata"]["chunk_index"],
                    "page": point.payload["metadata"].get("page"),
                    "text_preview": point.payload["text"][:160],
                }
                for point in points
            ],
        }

    def ask(self, question: str, top_k: int | None = None) -> dict:
        """Run the query pipeline: retrieve context first, then answer."""

        if not self.qdrant.collection_exists(self.settings.qdrant_collection_name):
            raise ValueError(
                "The Qdrant collection does not exist yet. Call POST /index first."
            )

        # LangChain is only used here as a small retrieval convenience layer.
        # Qdrant still stores the data and performs the nearest-neighbor search.
        vector_store = QdrantVectorStore(
            collection_name=self.settings.qdrant_collection_name,
            client=self.qdrant,
            embedding=self.langchain_embeddings,
            content_payload_key="text",
            metadata_payload_key="metadata",
        )
        results = vector_store.similarity_search_with_score(
            query=question,
            k=top_k or self.settings.top_k,
        )

        retrieved_chunks = [
            {
                "score": float(score),
                "text": document.page_content,
                "metadata": {
                    "source": document.metadata["source"],
                    "chunk_index": document.metadata["chunk_index"],
                    "page": document.metadata.get("page"),
                },
            }
            for document, score in results
        ]

        prompt = self._build_prompt(question, retrieved_chunks)
        answer = self._generate_answer(question, retrieved_chunks, prompt)

        return {
            "answer": answer,
            "prompt_used": prompt,
            "retrieved_chunks": retrieved_chunks,
        }

    def _load_documents(self, folder: Path) -> list:
        """Load supported files from a folder into LangChain Document objects."""

        if not folder.exists():
            raise FileNotFoundError(f"Document folder does not exist: {folder}")

        documents = []
        for file_path in sorted(folder.iterdir()):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            loader = self._get_loader(file_path)
            documents.extend(loader.load())

        if not documents:
            raise ValueError(
                f"No supported documents found in {folder}. "
                f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        return documents

    def _get_loader(self, file_path: Path):
        """Pick the simplest loader based on file extension."""

        if file_path.suffix.lower() == ".pdf":
            return PyPDFLoader(str(file_path))
        return TextLoader(str(file_path), encoding="utf-8")

    def _split_documents(self, documents: Iterable) -> list:
        """Split documents into smaller chunks and attach clean metadata."""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(list(documents))

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["source"] = Path(chunk.metadata["source"]).name
            if "page" in chunk.metadata:
                chunk.metadata["page"] = int(chunk.metadata["page"]) + 1

        return chunks

    def _build_points(self, chunks: list) -> list[PointStruct]:
        """Convert chunks into Qdrant points: id + vector + payload."""

        chunk_texts = [chunk.page_content for chunk in chunks]
        vectors = self.langchain_embeddings.embed_documents(chunk_texts)

        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        # ``text`` is the chunk body returned later to the user.
                        "text": chunk.page_content,
                        "metadata": {
                            "source": chunk.metadata["source"],
                            "chunk_index": chunk.metadata["chunk_index"],
                            "page": chunk.metadata.get("page"),
                        },
                    },
                )
            )
        return points

    def _recreate_collection(self) -> None:
        """Create a fresh Qdrant collection configured for cosine search."""

        vector_size = self.embedder.get_sentence_embedding_dimension()

        if self.qdrant.collection_exists(self.settings.qdrant_collection_name):
            self.qdrant.delete_collection(self.settings.qdrant_collection_name)

        self.qdrant.create_collection(
            collection_name=self.settings.qdrant_collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def _build_prompt(self, question: str, retrieved_chunks: list[dict]) -> str:
        """Build the final prompt using only retrieved context."""

        context = "\n\n".join(
            [
                (
                    f"Source: {chunk['metadata']['source']} | "
                    f"Chunk: {chunk['metadata']['chunk_index']} | "
                    f"Page: {chunk['metadata'].get('page')}\n"
                    f"{chunk['text']}"
                )
                for chunk in retrieved_chunks
            ]
        )

        return (
            "Answer the question using only the context below.\n"
            "If the context does not contain enough information, say that clearly.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

    def _generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict],
        prompt: str,
    ) -> str:
        """Generate an answer while staying conservative about weak retrieval."""

        if not retrieved_chunks:
            return "I could not answer because no relevant chunks were retrieved from Qdrant."

        best_score = max(chunk["score"] for chunk in retrieved_chunks)
        if best_score < self.settings.similarity_threshold:
            return (
                "I could not answer confidently from the retrieved context. "
                "The similarity scores were too low to support a grounded answer."
            )

        generation = self.answer_generator(prompt)[0]["generated_text"].strip()
        if not generation:
            return "I could not produce an answer from the retrieved context."

        if generation.lower() == question.lower():
            return (
                "I could not answer clearly from the retrieved context. "
                "Please inspect the retrieved chunks directly."
            )

        return generation
