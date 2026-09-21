import argparse

from lib.hybrid_search import HybridSearch
from lib.keyword_search import load_movies
from lib.llm import request_to_llm
from lib.semantic_search import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parse = subparsers.add_parser(
        "summarize", help="summarize retrieved queries"
    )

    summarize_parse.add_argument("query", type=str, help="Search query for RAG")
    summarize_parse.add_argument(
        "--limit", type=int, default=5, help="Search query for RAG"
    )

    citation_parse = subparsers.add_parser("citations", help="citate retrieved queries")

    citation_parse.add_argument("query", type=str, help="Search query for RAG")
    citation_parse.add_argument(
        "--limit", type=int, default=5, help="Search query for RAG"
    )

    question_parse = subparsers.add_parser("question", help="citate retrieved queries")

    question_parse.add_argument("question", type=str, help="Search query for RAG")
    question_parse.add_argument(
        "--limit", type=int, default=5, help="Search query for RAG"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))

            hybrid = HybridSearch(documents["movies"])

            result = hybrid.rrf_search(query, limit=args.limit)
            retrieved_title = []
            for res in result:
                retrieved_title.append(
                    f"{hybrid.idx.docmap[res['doc_id']]['title']} {hybrid.idx.docmap[res['doc_id']]['description']}"
                )

            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
            Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
            Provide a comprehensive answer that addresses the user's query.

            Query: {query}

            Documents:
            {retrieved_title}

            Answer:"""

            answer = request_to_llm(prompt)
            print("Search Results:")
            for res in result:
                print(f" - {hybrid.idx.docmap[res['doc_id']]['title']}")

            print("RAG Response:")
            print(answer)
        case "summarize":
            query = args.query

            documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))

            hybrid = HybridSearch(documents["movies"])

            result = hybrid.rrf_search(query, limit=args.limit)
            retrieved_title = []
            for res in result:
                retrieved_title.append(
                    f"{hybrid.idx.docmap[res['doc_id']]['title']} {hybrid.idx.docmap[res['doc_id']]['description']}"
                )

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

            This should be tailored to Webflyx users. Webflyx is a movie streaming service.

            Query: {query}

            Search results:
            {retrieved_title}

            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

            answer = request_to_llm(prompt)
            print("Search Results:")
            for res in result:
                print(f" - {hybrid.idx.docmap[res['doc_id']]['title']}")

            print("RAG Response:")
            print(answer)

        case "citations":
            query = args.query

            documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))

            hybrid = HybridSearch(documents["movies"])

            result = hybrid.rrf_search(query, limit=args.limit)
            retrieved_title = []
            for res in result:
                retrieved_title.append(
                    f"{hybrid.idx.docmap[res['doc_id']]['title']} {hybrid.idx.docmap[res['doc_id']]['description']}"
                )

            prompt = f"""Answer the query below and give information based on the provided documents.

            The answer should be tailored to users of Webflyx, a movie streaming service.
            If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

            Query: {query}

            Documents:
            {retrieved_title}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources in the format [1], [2], etc. when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the provided documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""

            answer = request_to_llm(prompt)
            print("Search Results:")
            for res in result:
                print(f" - {hybrid.idx.docmap[res['doc_id']]['title']}")

            print("RAG Response:")
            print(answer)
        case "question":
            query = args.question

            documents = load_movies(str(PROJECT_ROOT / "data" / "movies.json"))

            hybrid = HybridSearch(documents["movies"])

            result = hybrid.rrf_search(query, limit=args.limit)
            retrieved_title = []
            for res in result:
                retrieved_title.append(
                    f"{hybrid.idx.docmap[res['doc_id']]['title']} {hybrid.idx.docmap[res['doc_id']]['description']}"
                )

            prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

            Question: {query}

            Documents:
            {retrieved_title}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""

            answer = request_to_llm(prompt)
            print("Search Results:")
            for res in result:
                print(f" - {hybrid.idx.docmap[res['doc_id']]['title']}")

            print("Answer:")
            print(answer)
        case "":
            parser.print_help()


if __name__ == "__main__":
    main()
