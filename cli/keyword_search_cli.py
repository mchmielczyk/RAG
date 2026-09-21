import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from cli.lib.keyword_search import BM25_B, BM25_K1, InvertedIndex, tokenize_text_single


def build_command():
    inv_index = InvertedIndex()
    inv_index.build()
    inv_index.save()


def bm25_idf_command(term: str) -> float:
    inv_index = InvertedIndex()
    inv_index.load()
    tokenized_term = tokenize_text_single(term)
    return inv_index.get_bm25_idf(tokenized_term)


def bm25_tf_command(
    doc_id: int, term: str, k1: int = BM25_K1, b: int = BM25_B
) -> float:
    inv_index = InvertedIndex()
    inv_index.load()
    tokenized_term = tokenize_text_single(term)
    return inv_index.get_bm25_tf(doc_id, tokenized_term, k1, b)


def bm25_command(query, limit=5, k1: int = BM25_K1, b: int = BM25_B):
    inv_index = InvertedIndex()
    inv_index.load()
    result = inv_index.bm25_search(query, limit)
    for id, res in result.items():
        print(f"{id} {inv_index.docmap[id]['title']} - Score: {res:.2f}")


def main() -> None:

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Tokenize")
    tf_parser.add_argument("doc_id", type=int)
    tf_parser.add_argument("term", type=str)

    idf_parser = subparsers.add_parser("idf", help="Inverse document filtering")
    idf_parser.add_argument("term", type=str)

    tfidf_parser = subparsers.add_parser("tfidf", help="tfidf")
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term", type=str)

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print("Searching for: " + args.query)

            inv_index = InvertedIndex()
            inv_index.load()
            list_of_queries = list()
            list_of_queries = inv_index.get_documents(args.query)
            for id, movie in list_of_queries.items():
                print(f"{movie['title']} {movie['id']}")
        case "build":
            build_command()
        case "tf":
            single_token = tokenize_text_single(args.term)
            inv_index = InvertedIndex()
            inv_index.load()
            print(inv_index.get_tf(args.doc_id, single_token))
        case "idf":
            single_token = tokenize_text_single(args.term)
            inv_index = InvertedIndex()
            inv_index.load()
            print(
                f"Inverse document frequency of '{args.term}': {inv_index.get_idf(single_token):.2f}"
            )
        case "tfidf":
            single_token = tokenize_text_single(args.term)
            inv_index = InvertedIndex()
            inv_index.load()
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {inv_index.get_tfidf(args.doc_id, args.term):.2f}"
            )
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            bm25_command(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
