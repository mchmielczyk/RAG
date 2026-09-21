import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def normalize(set):
    num_set = []
    max_num = max(set)
    min_num = min(set)
    for num in set:
        if min_num == max_num:
            num_set.append(1.0)
        else:
            num_set.append((num - min_num) / (max_num - min_num))

    return num_set


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:

        bm25_result = self._bm25_search(query, limit * 500)
        semantic_chunked_result = self.semantic_search.search_chunks(query, limit * 500)
        semantic_chunked = []
        for score in semantic_chunked_result:
            semantic_chunked.append({"id": score["id"], "score": score["score"]})

        normalized_bm25 = {
            result: {"id": result, "score": None} for result in bm25_result
        }
        bm25_result = normalize(bm25_result.values())
        for res in zip(normalized_bm25, bm25_result):
            normalized_bm25[res[0]]["score"] = res[1]

        normalized_semantic = {
            result["id"]: {"id": result["id"], "score": None}
            for result in semantic_chunked
        }
        scores = [result["score"] for result in semantic_chunked]
        semantic_chunked = normalize(scores)
        for res in zip(normalized_semantic, semantic_chunked):
            normalized_semantic[res[0]]["score"] = res[1]

        hybrid_map = {}
        for res in normalized_bm25.values():
            if res["id"] in normalized_semantic:
                semantic_score = normalized_semantic[res["id"]]["score"]
            else:
                semantic_score = 0
            hybrid_map[res["id"]] = {
                "doc_id": res["id"],
                "keyword_score": res["score"],
                "semantic_score": semantic_score,
                "hybrid_score": hybrid_score(res["score"], semantic_score, alpha),
            }

        hybrid_map = sorted(
            hybrid_map.values(), key=lambda item: item["hybrid_score"], reverse=True
        )
        return hybrid_map[:limit]

    def rrf_search(self, query: str, k: int = 60, limit: int = 10) -> list[dict]:

        bm25_result = self._bm25_search(query, limit * 500)
        semantic_chunked_result = self.semantic_search.search_chunks(query, limit * 500)
        semantic_chunked = []
        for score in semantic_chunked_result:
            semantic_chunked.append({"id": score["id"], "score": score["score"]})

        normalized_bm25 = {
            result: {"id": result, "score": None} for result in bm25_result
        }
        bm25_result = sorted(
            bm25_result.items(), key=lambda item: item[1], reverse=True
        )
        bm25_ranked = {
            result[0]: {"id": result[0], "score": i}
            for i, result in enumerate(bm25_result, 1)
        }
        bm25_ranked = dict(sorted(bm25_ranked.items(), key=lambda item: item[0]))
        for res in zip(normalized_bm25, bm25_ranked):
            normalized_bm25[res[0]]["score"] = bm25_ranked[res[0]]["score"]

        normalized_semantic = {
            result["id"]: {"id": result["id"], "score": None}
            for result in semantic_chunked
        }
        semantic_chunked = sorted(
            semantic_chunked, key=lambda item: item["score"], reverse=True
        )
        semantic_ranked = {
            result["id"]: {"id": result["id"], "score": i}
            for i, result in enumerate(semantic_chunked, 1)
        }
        semantic_ranked = dict(
            sorted(semantic_ranked.items(), key=lambda item: item[0])
        )
        for res in zip(normalized_semantic, semantic_ranked):
            normalized_semantic[res[0]]["score"] = semantic_ranked[res[0]]["score"]

        hybrid_map = {}
        for res in normalized_bm25.values():
            if res["id"] in normalized_semantic:
                semantic_score = normalized_semantic[res["id"]]["score"]
            else:
                semantic_score = len(normalized_semantic)
            hybrid_map[res["id"]] = {
                "doc_id": res["id"],
                "keyword_score": res["score"],
                "semantic_score": semantic_score,
                "rrf_score": rrf_score(res["score"], k) + rrf_score(semantic_score, k),
            }

        hybrid_map = sorted(
            hybrid_map.values(), key=lambda item: item["rrf_score"], reverse=True
        )
        return hybrid_map[:limit]
