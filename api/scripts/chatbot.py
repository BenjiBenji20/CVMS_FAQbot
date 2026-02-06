from api.config.settings import settings
from api.scripts.vector_store import vector_store
from groq import Groq

# Initialize Groq client
llm = Groq(api_key=settings.LLM_API_KEY)

# Set up vector store as retriever
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

def chatbot(message: str) -> str:
    """
    Build LLM prompt along with user's query and extracted knowledge 
    using the retriever.
    
    Args:
        message: User's question
    
    Yields:
        str: Partial responses as they're generated
    """
    # Retrieve relevant chunks
    docs = retriever.invoke(message)
    
    # Build knowledge base
    knowledge = "\n\n".join([doc.page_content for doc in docs])
    
    # Build messages for Groq
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that answers questions based on provided knowledge.
            You rely solely on the information in the knowledge section provided to you.
            Answer naturally without mentioning that you're using provided knowledge."""
        },
        {
            "role": "user",
            "content": f"""Based on the following knowledge, please answer the question.

            Knowledge:
            {knowledge}

            Question: {message}"""
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
