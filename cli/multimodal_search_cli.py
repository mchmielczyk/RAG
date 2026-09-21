import argparse

from lib.keyword_search import load_movies
from lib.multimodal_search import image_search_command, verify_image_embedding
from lib.semantic_search import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal search")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "verify_image_embedding", help="verify imege embeddings"
    )
    rag_parser.add_argument("path", type=str, help="parh to image file")

    search = subparsers.add_parser(
        "image_search", help="search database with a help of image"
    )
    search.add_argument("path", type=str, help="parh to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case "image_search":
            documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))[
                "movies"
            ]
            image_search_command(args.path, documents)


if __name__ == "__main__":
    main()
