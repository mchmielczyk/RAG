# before#{'chunk_idx': self.chunk_metadata[i]['chunk_idx'],'movie_idx':self.chunk_metadata[i]['movie_idx'],'score':cosine_similarity(embedded_query,chunk)}
def format_search_result(
    before_processing: list[dict],
    document_map,
    chunk_metadata,
    SCORE_PRECISION: int = 2,
):

    result = []
    for input in before_processing:
        idx = input[0]
        result.append(
            {
                "id": idx,
                "title": document_map[idx]["title"],
                "document": str(document_map[idx])[:100],
                "score": round(input[1], SCORE_PRECISION),
                "metadata": chunk_metadata[input[0]] or {},
            }
        )

    return result
