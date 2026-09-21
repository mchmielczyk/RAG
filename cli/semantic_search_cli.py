import argparse


def main() -> None:

    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    sub_parsers = parser.add_subparsers(dest="command", help="Available commands")
    sub_parsers.add_parser("verify", help="verify model")

    embed_parser = sub_parsers.add_parser(
        "embed_text", help="create embedding of passed text"
    )
    embed_parser.add_argument("text", type=str)

    sub_parsers.add_parser("verify_embeddings", help="verify embeddings")

    embed_query_parser = sub_parsers.add_parser(
        "embed_query", help="create embedding of passed query"
    )
    embed_query_parser.add_argument("query", type=str)

    search_parser = sub_parsers.add_parser(
        "search", help="semantic search in documents"
    )
    search_parser.add_argument("query", type=str)
    search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="limit number of results"
    )

    chunk_parser = sub_parsers.add_parser("chunk", help="split text into chunks")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument(
        "--chunk-size", type=int, nargs="?", default=200, help="size of text chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="set overlap in chunk"
    )

    semantic_chunk = sub_parsers.add_parser(
        "semantic_chunk", help="split text into semantic chunks"
    )
    semantic_chunk.add_argument("text", type=str)
    semantic_chunk.add_argument(
        "--max-chunk-size",
        type=int,
        nargs="?",
        default=4,
        help="max size of text chunk",
    )
    semantic_chunk.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="set overlap in chunk"
    )

    embed_chunk = sub_parsers.add_parser(
        "embed_chunks", help="embed chunks from json file"
    )

    search_chunk = sub_parsers.add_parser(
        "search_chunked", help="search semantic query in chunked env"
    )
    search_chunk.add_argument("query", type=str)
    search_chunk.add_argument(
        "--limit", type=int, nargs="?", default=5, help="limit number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "":
            parser.print_help()
        case "verify":
            from lib.semantic_search import verify_model

            verify_model()
        case "embed_text":
            from lib.semantic_search import embed_text

            embed_text(args.text)
        case "verify_embeddings":
            from lib.semantic_search import verify_embeddings

            verify_embeddings()
        case "embed_query":
            from lib.semantic_search import embed_query_text

            embed_query_text(args.query)
        case "search":
            from lib.keyword_search import load_movies
            from lib.semantic_search import PROJECT_ROOT, SemanticSearch

            semantic_search = SemanticSearch()
            documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))
            semantic_search.load_or_create_embeddings(documents["movies"])
            result = semantic_search.search(args.query, args.limit)
            for res in result:
                print(f"{res['title']} (score: {res['score']})\n{res['description']}")
        case "chunk":
            from lib.semantic_search import chunk

            result = chunk(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            i = 1
            for res in result:
                print(f"{i}. {res}")
                i += 1
        case "semantic_chunk":
            from lib.semantic_search import semantic_chunk

            result = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            i = 1
            for res in result:
                print(f"{i}. {res}")
                i += 1
        case "embed_chunks":
            from lib.keyword_search import load_movies
            from lib.semantic_search import PROJECT_ROOT, ChunkedSemanticSearch

            documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))
            chunked = ChunkedSemanticSearch()
            embeddings = chunked.load_or_create_chunk_embeddings(documents["movies"])
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            from lib.keyword_search import load_movies
            from lib.semantic_search import PROJECT_ROOT, ChunkedSemanticSearch

            documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))
            chunked = ChunkedSemanticSearch()
            embeddings = chunked.load_or_create_chunk_embeddings(documents["movies"])

            result = chunked.search_chunks(args.query, args.limit)

            for i, res in enumerate(result):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")


if __name__ == "__main__":
    main()
