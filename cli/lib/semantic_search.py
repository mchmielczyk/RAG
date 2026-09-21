import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / "cache" / "movie_embeddings.npy"
CACHE_PATH_SEMANTIC_EMBEDDINGS = PROJECT_ROOT / "cache" / "chunk_embeddings.npy"
CACHE_PATH_SEMANTIC_EMBEDDINGS_JSON = PROJECT_ROOT / "cache" / "chunk_metadata.json"
from lib.keyword_search import load_movies
from lib.search_utils import format_search_result


def chunk(text: str, size: int, overlap: int) -> list[str]:
    splitted_text = text.split()
    result = []
    for i in range(0, len(splitted_text), size):
        if i - overlap < 0:
            result.append(" ".join(splitted_text[0 : i + size]))
        else:
            result.append(" ".join(splitted_text[i - overlap : i + size - overlap]))

    return result


def semantic_chunk(text: str, size: int, overlap: int) -> list[str]:
    splitted_text = text.strip()
    if splitted_text is None:
        return []

    if (
        len(splitted_text) == 1
        and str(splitted_text).endswith([".", "!", "?"]) == False
    ):
        return splitted_text

    splitted_text = re.split(r"(?<=[.!?])\s+", splitted_text)
    result = []
    step = size - overlap
    for i in range(0, len(splitted_text), step):
        if i - overlap < 0:
            text_to_append = " ".join(splitted_text[0:size])
            text_to_append.strip()
            if text_to_append is None:
                continue
            result.append(text_to_append)
        else:
            text = splitted_text[i : i + size]
            if len(text) <= overlap:
                break
            else:
                text_to_append = " ".join(text)
                text_to_append.strip()
                if text_to_append is None:
                    continue
                result.append(text_to_append)

    return result


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_model():
    semanticSearch = SemanticSearch()
    print(f"Model loaded: {semanticSearch.model}")
    print(f"Max sequence length: {semanticSearch.model.max_seq_length}")


def verify_embeddings():
    semantic_search = SemanticSearch()

    documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))
    documents = documents["movies"]

    embeddings = semantic_search.load_or_create_embeddings(documents)


def embed_text(text):
    semanticSearch = SemanticSearch()
    embedding = semanticSearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query):
    semanticSearch = SemanticSearch()
    embedding = semanticSearch.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text == "" or text.isspace():
            raise ValueError("String is empty or have only whitespaces")

        embeddings = self.model.encode([text])

        return embeddings[0]

    def build_embeddings(self, documents):

        document_texts = [
            f"{document['title']}: {document['description']}" for document in documents
        ]

        self.embeddings = self.model.encode(
            document_texts,
            show_progress_bar=True,
        )

        with open(CACHE_PATH, "wb") as file:
            np.save(file, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {document["id"]: document for document in documents}

        if CACHE_PATH.exists():
            with open(CACHE_PATH, "rb") as file:
                self.embeddings = np.load(file)

        if self.embeddings is not None and self.embeddings.shape[0] == len(
            self.documents
        ):
            return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarity_ranking = []
        for doc_e, doc in zip(self.embeddings, self.documents):
            similarity_ranking.append([cosine_similarity(query_embedding, doc_e), doc])

        sorted_by_relativity = sorted(
            similarity_ranking, key=lambda value: value[0], reverse=True
        )

        sorted_by_relativity = sorted_by_relativity[:limit]
        result = list()
        for index in sorted_by_relativity:
            result.append(
                {
                    "score": index[0],
                    "title": index[1]["title"],
                    "description": index[1]["description"],
                }
            )
        return result


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:

        self.documents = documents

        for document in documents:
            self.document_map[document["id"]] = document

        all_chunks = []
        metadata = []

        for document in documents:
            if document["description"] is None:
                continue
            chunks = semantic_chunk(document["description"], 4, 1)
            all_chunks.extend(chunks)
            i = 1
            for ch in chunks:
                metadata.append(
                    {
                        "movie_idx": document["id"],
                        "chunk_idx": i,
                        "total_chunks": len(chunks),
                    }
                )
                i += 1

        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = metadata

        with open(CACHE_PATH_SEMANTIC_EMBEDDINGS, "wb") as f:
            np.save(f, self.chunk_embeddings)

        with open(CACHE_PATH_SEMANTIC_EMBEDDINGS_JSON, "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:

        self.documents = documents

        for document in documents:
            self.document_map[document["id"]] = document

        if (
            CACHE_PATH_SEMANTIC_EMBEDDINGS.exists()
            and CACHE_PATH_SEMANTIC_EMBEDDINGS_JSON.exists()
        ):
            with open(CACHE_PATH_SEMANTIC_EMBEDDINGS, "rb") as file:
                self.chunk_embeddings = np.load(file)

            with open(CACHE_PATH_SEMANTIC_EMBEDDINGS_JSON, "r") as file:
                data = json.load(file)
                self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings

        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        embedded_query = self.generate_embedding(query)

        chunk_score = []

        for i, chunk in enumerate(self.chunk_embeddings):
            chunk_score.append(
                {
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": cosine_similarity(embedded_query, chunk),
                }
            )

        movie_map = {}

        for score in chunk_score:
            if (score["movie_idx"] not in movie_map) or (
                score["score"] > movie_map[score["movie_idx"]]
            ):
                movie_map[score["movie_idx"]] = score["score"]

        movie_map = sorted(movie_map.items(), key=lambda item: item[1], reverse=True)

        movie_map = movie_map[:limit]

        result = format_search_result(movie_map, self.document_map, self.chunk_metadata)

        return result
