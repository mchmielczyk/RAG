from lib.semantic_search import cosine_similarity
from PIL import Image
from sentence_transformers import SentenceTransformer


def verify_image_embedding(path):
    multimodal = MultimodalSearch()
    embedding = multimodal.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(path, documents):
    multimodal_search = MultimodalSearch(documents)
    results = multimodal_search.search_with_image(path)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (similarity: {res['score']:.3})")
        print(f"{res['description'][:100]}...")


class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.sentence_transformer = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = self.__concatenate_documents(self.documents)
        self.text_embeddings = self.sentence_transformer.encode(
            self.texts, show_progress_bar=True
        )

    def embed_image(self, path):
        image_file = Image.open(path)

        return self.sentence_transformer.encode(image_file)

    def __concatenate_documents(self, documents):
        all_documents = []
        for doc in documents:
            all_documents.append(f"{doc['title']}: {doc['description']}")

        return all_documents

    def search_with_image(self, path):
        image_embedding = self.embed_image(path)

        cosine_similarity_results = []

        for i, doc in enumerate(self.text_embeddings, 0):
            cosine_similarity_results.append(
                {
                    "doc_id": i,
                    "title": self.documents[i]["title"],
                    "description": self.documents[i]["description"],
                    "score": cosine_similarity(image_embedding, doc),
                }
            )

        return sorted(
            cosine_similarity_results, key=lambda x: x["score"], reverse=True
        )[:5]
