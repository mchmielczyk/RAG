import argparse
import json

from lib.hybrid_search import HybridSearch


def load_golden(path: str) -> dict:
    with open(path, "r") as golden:
        golden_database = dict(json.load(golden))
    return golden_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    dataset = load_golden("data/golden_dataset.json")["test_cases"]
    from lib.keyword_search import load_movies
    from lib.semantic_search import PROJECT_ROOT

    documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))

    hybrid = HybridSearch(documents["movies"])

    print(f"k={limit}")
    for test_set in dataset:
        query = test_set["query"]
        relevant = test_set["relevant_docs"]

        relevant_retrieved = hybrid.rrf_search(query, 60, limit)

        retrieved_movies = []
        precision_total = 0
        for res in relevant_retrieved:
            movie = hybrid.idx.docmap[res["doc_id"]]["title"]
            retrieved_movies.append(movie)
            if movie in relevant:
                precision_total += 1

        precision = (precision_total / limit) if limit > 0 else 0
        recall = precision_total / len(relevant)

        if precision + recall == 0:
            f1score = 0
        else:
            f1score = 2 * (precision * recall) / (precision + recall)

        print(f" - Query: {query}")
        print(f"     - Precision@{limit}: {precision:.4f}")
        print(f"     - Recall@{limit}: {recall:.4f}")
        print(f"     - F1 Score: {f1score:.4f}")
        print(f"     - Retrieved: {str.join(',', retrieved_movies)}")
        print(f"     - Relevant: {str.join(',', relevant)}")

    # run evaluation logic here


if __name__ == "__main__":
    main()
