import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    sub_parsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = sub_parsers.add_parser("normalize", help="normalize set of numbers")
    normalize.add_argument("numbers", nargs="*", type=float)

    weighted = sub_parsers.add_parser(
        "weighted-search", help="weighted search query in chunked env"
    )
    weighted.add_argument("query", type=str)
    weighted.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        default=0.5,
        help="set beetween keywoard seach and semantic",
    )
    weighted.add_argument(
        "--limit", type=int, nargs="?", default=5, help="limit number of results"
    )
    rrf = sub_parsers.add_parser(
        "rrf-search", help="weighted search query in chunked env"
    )
    rrf.add_argument("query", type=str)
    rrf.add_argument("--k", type=int, nargs="?", default=60, help="step the curve of k")
    rrf.add_argument(
        "--limit", type=int, nargs="?", default=5, help="limit number of results"
    )
    rrf.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Query reranking method",
    )
    rrf.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        help="evaluate by llm",
    )

    args = parser.parse_args()

    match args.command:
        case "":
            parser.print_help()
        case "normalize":
            if args.numbers is None:
                return
            num_set = []
            max_num = max(args.numbers)
            min_num = min(args.numbers)
            for num in args.numbers:
                if min_num == max_num:
                    num_set.append(1.0)
                else:
                    num_set.append((num - min_num) / (max_num - min_num))

            for score in num_set:
                print(f"* {score:.4f}")
        case "weighted-search":
            from lib.keyword_search import load_movies
            from lib.semantic_search import PROJECT_ROOT

            documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))

            from lib.hybrid_search import HybridSearch

            hybrid = HybridSearch(documents["movies"])

            result = hybrid.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(result, 1):
                print(f"{i}. {hybrid.idx.docmap[result['doc_id']]['title']}")
                print(f"Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"BM25: {result['keyword_score']:.3f} Semantic:{result['semantic_score']:.3f}"
                )
                print(f"{hybrid.idx.docmap[result['doc_id']]['description'][:100]}...")

        case "rrf-search":
            logger = logging.getLogger("rrf_search_logger")
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                file_handler = logging.FileHandler(
                    "hybrid_search.log", encoding="utf-8"
                )
                logger.addHandler(file_handler)

            logger.propagate = False

            query = args.query
            logger.info(f"original query: {query}")
            if args.enhance == "spell":
                from lib.llm import request_to_llm

                enhanced_query = f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{args.query}"
                """
                enhanced_query = request_to_llm(enhanced_query)
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query
            elif args.enhance == "rewrite":
                from lib.llm import request_to_llm

                enhanced_query = f"""Rewrite the user-provided movie search query below to be more specific and searchable.
                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{args.query}"
                """
                enhanced_query = request_to_llm(enhanced_query)
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query
            elif args.enhance == "expand":
                from lib.llm import request_to_llm

                enhanced_query = f"""Expand the user-provided movie search query below with related terms.
                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.
                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"
                User query: "{args.query}"
                """
                enhanced_query = request_to_llm(enhanced_query)
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query

            logger.info(f"enhanced query: {query}")

            from lib.keyword_search import load_movies
            from lib.semantic_search import PROJECT_ROOT

            documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))

            from lib.hybrid_search import HybridSearch

            hybrid = HybridSearch(documents["movies"])

            if args.rerank_method == "individual":
                limit = args.limit * 5
            else:
                limit = args.limit
            result = hybrid.rrf_search(query, args.k, limit)

            logger_result = hybrid.rrf_search(query, args.k, limit * 5)

            logger.info("results from rrf_search by score:")
            for i, res in enumerate(logger_result, 1):
                logger.info(f"{i}. {hybrid.idx.docmap[res['doc_id']]['title']}")

            if args.rerank_method == "individual":
                import time

                from lib.llm import request_to_llm

                rerank_results = []
                for doc in result:
                    movie = hybrid.idx.docmap[doc["doc_id"]]
                    message = f"""Rate how well this movie matches the search query.

                    Query: "{query}"
                    Movie: {movie["title"]} - {movie["description"]}

                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                    Rate 0-10 (10 = perfect match).
                    Output ONLY the number in your response, no other text or explanation.
                    For example "5" and that is all no more output text

                    Score:"""
                    response = request_to_llm(message)
                    # print(response)
                    # print(int(response))
                    try:
                        score = int(response.strip())
                        if not 0 <= score <= 10:
                            continue
                        rerank_results.append({"score": score, "document": doc})
                    except ValueError:
                        continue
                    time.sleep(15)
                rerank_results = sorted(
                    rerank_results, key=lambda res: res["score"], reverse=True
                )[: args.limit]

                for i, res in enumerate(rerank_results, 1):
                    logger.info(f"results from re-rank: {args.rerank_method} by score:")
                    logger.info(
                        f"{i}. {hybrid.idx.docmap[res['document']['doc_id']]['title']}"
                    )

                for i, result in enumerate(rerank_results, 1):
                    print(
                        f"{i}. {hybrid.idx.docmap[result['document']['doc_id']]['title']}"
                    )
                    print(f"Re-rank Score: {result['score']:.3f}/10")
                    print(f"Hybrid Score: {result['document']['rrf_score']:.3f}")
                    print(
                        f"BM25 Rank: {result['document']['keyword_score']}, Semantic Rank:{result['document']['semantic_score']}"
                    )
                    print(
                        f"{hybrid.idx.docmap[result['document']['doc_id']]['description'][:100]}..."
                    )

                return

            elif args.rerank_method == "batch":
                import json
                import time

                from lib.llm import request_to_llm

                movie = ""
                rerank_results = []
                for i, doc in enumerate(result, 1):
                    movie += f"title {i}: {hybrid.idx.docmap[doc['doc_id']]['title']} description {i}: {hybrid.idx.docmap[doc['doc_id']]['description']}"

                message = f"""Rank the movies listed below by relevance to the following search query.

                Query: "{query}"

                Movies:
                {movie}

                Return the movie IDs in order of relevance, best match first.

                Your response must be a raw JSON array of integers.
                Do not wrap the JSON in Markdown. Do not use a ```json code block.
                Do not include any explanatory text.

                For example:
                [75, 12, 34, 2, 1]

                Ranking:"""

                response = request_to_llm(message)

                rerank = json.loads(response)
                for doc in zip(result, rerank):
                    score = 0
                    try:
                        score = doc[1]
                        if not 0 <= score <= 10:
                            continue
                    except ValueError:
                        continue

                    rerank_results.append({"score": score, "document": doc[0]})

                rerank_results = sorted(
                    rerank_results, key=lambda res: res["score"], reverse=True
                )[: args.limit]

                for i, res in enumerate(rerank_results, 1):
                    logger.info(f"results from re-rank: {args.rerank_method} by score:")
                    logger.info(
                        f"{i}. {hybrid.idx.docmap[res['document']['doc_id']]['title']}"
                    )

                print(f"Re-ranking top {args.limit} results using batch method...")
                for i, result in enumerate(rerank_results, 1):
                    print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")
                    print(
                        f"{i}. {hybrid.idx.docmap[result['document']['doc_id']]['title']}"
                    )
                    print(f"Re-rank Score: {result['score']:.3f}/10")
                    print(f"Hybrid Score: {result['document']['rrf_score']:.3f}")
                    print(
                        f"BM25 Rank: {result['document']['keyword_score']}, Semantic Rank:{result['document']['semantic_score']}"
                    )
                    print(
                        f"{hybrid.idx.docmap[result['document']['doc_id']]['description'][:100]}..."
                    )

                return

            elif args.rerank_method == "cross_encoder":
                from sentence_transformers import CrossEncoder

                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

                pairs = []
                for doc in result:
                    pairs.append(
                        [
                            query,
                            f"{hybrid.idx.docmap[doc['doc_id']]['title']} - {hybrid.idx.docmap[doc['doc_id']]['description']}",
                        ]
                    )

                score = cross_encoder.predict(pairs)
                rerank_results = []
                for res in zip(score, result):
                    rerank_results.append({"score": res[0], "document": res[1]})

                rerank_results = sorted(
                    rerank_results, key=lambda res: res["score"], reverse=True
                )[: args.limit]
                print(
                    f"Re-ranking top {args.limit} results using cross_encoder method..."
                )

                for i, res in enumerate(rerank_results, 1):
                    logger.info(f"results from re-rank: {args.rerank_method} by score:")
                    logger.info(
                        f"{i}. {hybrid.idx.docmap[res['document']['doc_id']]['title']}"
                    )

                for i, result in enumerate(rerank_results, 1):
                    print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")
                    print(
                        f"{i}. {hybrid.idx.docmap[result['document']['doc_id']]['title']}"
                    )
                    print(f"Cross Encoder Score: {result['score']:.3f}/10")
                    print(f"RRF Score: {result['document']['rrf_score']:.3f}")
                    print(
                        f"BM25 Rank: {result['document']['keyword_score']}, Semantic Rank:{result['document']['semantic_score']}"
                    )
                    print(
                        f"{hybrid.idx.docmap[result['document']['doc_id']]['description'][:100]}..."
                    )

                return
            result2 = result
            result3 = result
            for i, result in enumerate(result, 1):
                print(f"{i}. {hybrid.idx.docmap[result['doc_id']]['title']}")
                print(f"Hybrid Score: {result['rrf_score']:.3f}")
                print(
                    f"BM25 Rank: {result['keyword_score']}, Semantic Rank:{result['semantic_score']}"
                )
                print(f"{hybrid.idx.docmap[result['doc_id']]['description'][:100]}...")
            result_str = ""
            for i, result in enumerate(result2, 1):
                result_str += f"{i}. {hybrid.idx.docmap[result['doc_id']]['title']} {hybrid.idx.docmap[result['doc_id']]['description']}"

            if args.evaluate:
                import json

                from lib.llm import request_to_llm

                message = f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {result_str}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers other than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]"""

                response = request_to_llm(message)

                rerank = json.loads(response)
                evaluate = []
                for doc in zip(result3, rerank):
                    score = 0
                    try:
                        score = doc[1]
                        if not 0 <= score <= 3:
                            continue
                    except ValueError:
                        continue

                    evaluate.append({"score": score, "document": doc[0]})

                for i, eval in enumerate(evaluate, 1):
                    print(
                        f"{i}. {hybrid.idx.docmap[eval['document']['doc_id']]['title']}: {eval['score']}/3"
                    )


if __name__ == "__main__":
    main()
