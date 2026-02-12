from api.config.settings import settings
from api.scripts.vector_store import vector_store
from groq import Groq
import json

# Initialize Groq client
llm = Groq(api_key=settings.LLM_API_KEY)

# Set up vector store as retriever
retriever = vector_store.as_retriever(
    search_kwargs={
        'k': 5,
        'filter': {'doc_type': 'faq'}  # Only search FAQs
    }
)


def llm_message_rephraser( original_message: str) -> str:
    """
    Rephrase original message into more effective semantic search
    """
    rephrased_messages = [
        {
            "role": "system",
            "content": (
                "You are a search query optimizer for a retrieval system used in the Philippine marketplace context.\n\n"
                "Your task is to rewrite informal, shorthand, or vague user messages into a clearer and "
                "retrieval-friendly search query.\n\n"
                "STRICT RULES:\n"
                "- Interpret Philippine common marketplace shorthands (e.g., hm = how much, loc = location, etc...).\n"
                "- Preserve the original intent.\n"
                "- Preserve important nouns or service-related words if present.\n"
                "- Expand vague or short phrases into a complete search query.\n"
                "- Do NOT introduce new services, features, packages, or assumptions.\n"
                "- Do NOT invent specific service names.\n"
                "- Do NOT add marketing or descriptive words.\n"
                "- Keep terminology neutral and aligned with the original message.\n"
                "- Output ONLY one single improved search query.\n"
                "- No explanations.\n"
                "- No extra text.\n"
                "- No quotation marks."
            )
        },
        {
            "role": "user",
            "content": (
                "ORIGINAL MESSAGE:\n"
                f"{original_message}"
            )
        }
    ]
    
    return stream_response(rephrased_messages, 0.3)


def stream_response(messages: list[dict[str, str]], temperature: float = 0.5) -> str:
    """
    Stream the response using Groq's streaming API
    """
    stream = llm.chat.completions.create(
        model=settings.LLM_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        stream=True
    )
    
    # concat chunks as they arrive
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content

    return response


def chatbot(message: str, to_rephrase: bool = False) -> tuple[str, list[float]]:
    """
    Build LLM prompt along with client's query and extracted knowledge 
    using the retriever.
    
    Args:
        message: User's question
    
    Yields:
        str: Partial responses as they're generated
    """
    # Retrieve relevant chunks
    docs = retriever.vectorstore.similarity_search_with_score(message, k=5)
    
    # Filter by relevance threshold
    relevant_docs = [
        doc for doc, score in docs
        if score < 0.7
    ]
    
    relevant_scores = [
        score for _, score in docs
        if score < 0.7
    ]
    
    # if already done rephrase and still doesn't have relevant scores, return fallback
    if to_rephrase and len(relevant_scores) < 1:
        return (
                "I'm sorry, I couldn't find this information in our official documents. "
                "Please contact us via at cvmscustomerservice@gmail.com or visit our office for further assistance.\""
            ), []
    
    # Build knowledge base
    knowledge = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Build messages for Groq
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant for Colour Variant Multimedia Services.\n\n"
                "STRICT RULES:\n"
                "- You MUST answer using ONLY the information provided in the context.\n"
                "- EXCEPTION: If the user greets you (e.g., Hi, Hello, Good day), you may respond with a polite greeting "
                "without using the context.\n"
                "- Use at most one emoji, only if appropriate.\n"
                "- Keep answers short and clear.\n"
                "- Do NOT use general knowledge.\n"
                "- Do NOT guess, assume, or fabricate information.\n"
                "- If the user asks a question and the answer is NOT explicitly stated in the context, respond with:\n"
                "\"I'm sorry, I couldn't find this information in our official documents. "
                "Please contact us via at cvmscustomerservice@gmail.com or visit our office for further assistance.\""
            )
        },
        {
            "role": "user",
            "content": (
                "CONTEXT:\n"
                f"{knowledge}\n\n"
                "QUESTION:\n"
                f"{message}"
            )
        }
    ]
    
    return stream_response(messages), relevant_scores
