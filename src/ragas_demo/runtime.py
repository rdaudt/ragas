from openai import OpenAI

from ragas_demo.config import Settings
from ragas_demo.rag import OpenAIAnswerGenerator, RagService
from ragas_demo.store import OpenAIEmbedder, VectorIndex


def create_openai_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key.get_secret_value())


def create_vector_index(settings: Settings, client: OpenAI | None = None) -> VectorIndex:
    active_client = client or create_openai_client(settings)
    embedder = OpenAIEmbedder(active_client, settings.embedding_model)
    return VectorIndex(settings.state_dir / "chroma", embedder)


def create_rag_service(settings: Settings) -> RagService:
    client = create_openai_client(settings)
    index = create_vector_index(settings, client)
    generator = OpenAIAnswerGenerator(client, settings.answer_model)
    return RagService(index, generator, top_k=settings.top_k)

