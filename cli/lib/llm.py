import base64
import os

from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    # base_url="https://api.openai.com/v1",
    # base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key,
)


def request_to_llm(query: str, role: str = "user"):
    messages = [
        {
            "role": "user",
            "content": query,
        }
    ]
    response = client.chat.completions.create(
        model="openrouter/free", messages=messages
    )
    # response = client.chat.completions.create(model='gpt-4o-mini' ,messages=messages)
    # response = client.chat.completions.create(model="gemini-3.6-flash" ,messages=messages)
    return response.choices[0].message.content


def request_to_llm_multimodal(query: str, image, mime_type, role: str = "user"):
    image_prommpt = """
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    query:
    """

    data_url = f"data:{mime_type};base64,{base64.b64encode(image).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": image_prommpt},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": query.strip()},
            ],
        }
    ]
    response = client.chat.completions.create(
        model="openrouter/free", messages=messages
    )
    # response = client.chat.completions.create(model='gpt-4o-mini' ,messages=messages)
    # response = client.chat.completions.create(model="gemini-3.6-flash" ,messages=messages)
    return response
