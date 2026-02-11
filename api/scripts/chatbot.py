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

def chatbot(message: str) -> str:
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

    
    # Stream the response using Groq's streaming API
    stream = llm.chat.completions.create(
        model=settings.LLM_NAME,
        messages=messages,
        temperature=0.5,
        max_tokens=1024,
        stream=True
    )
    
    # concat chunks as they arrive
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content

    return response
