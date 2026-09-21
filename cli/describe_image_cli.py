import argparse
import mimetypes

from lib.llm import request_to_llm_multimodal


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal search")
    parser.add_argument(
        "--image", required=True, type=str, help="the path to an image file"
    )
    parser.add_argument(
        "--query", required=True, help="a text query to rewrite based on the image"
    )
    args = parser.parse_args()

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        image_readed = f.read()

    response = request_to_llm_multimodal(args.query, image_readed, mime)
    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
