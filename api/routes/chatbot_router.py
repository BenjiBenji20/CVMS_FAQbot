from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import asyncio
import logging

from api.scripts.chatbot import retriever, chatbot

logger = logging.getLogger(__name__)

chatbot_router = APIRouter(prefix="/api/chat-ai", tags=["chatbot"])

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="User's message")

class ChatResponse(BaseModel):
    role: str
    message: str
    created_at: datetime


@chatbot_router.get("/health-check")
async def health_check():
    """
    Check if the chatbot service is healthy
    """
    try:
        # Test the retriever
        test_docs = await asyncio.to_thread(retriever.invoke, "test")
        
        return {
            "status": "healthy",
            "vector_store": "connected",
            "documents_in_store": len(test_docs) > 0
        }
    except Exception as e:
        logger.error(f"Chatbot service health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "vector_store": "disconnected",
            "error": str(e)
        }


@chatbot_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - returns complete response from chatbot
    """
    try:
        # Call chatbot function with retry logic
        ai_response: str = ""
        max_attempts = 3
        message: str = request.message.strip()
        
        # if the message is only whitespace and has been strpped
        # make synthetic message.
        if message is None or message == "":
            return ChatResponse(
                role="assistant",
                message="It looks like the question didn’t come through. Could you please provide the question you’d like answered?",
                created_at=datetime.now(timezone.utc)
            )
            
        for attempt in range(1, max_attempts + 1):
            try:
                # run chatbot in thread 
                ai_response = await asyncio.to_thread(chatbot, message)
                
                # Validate response
                if ai_response and len(ai_response.strip()) > 0:
                    break
                    
                logger.warning(f"Empty response on attempt {attempt}")
                
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                if attempt == max_attempts:
                    raise
                await asyncio.sleep(1)  # Brief delay before retry
        
        if not ai_response or len(ai_response.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to get valid response from chatbot after 3 attempts"
            )
        
        return ChatResponse(
            role="assistant",
            message=ai_response,
            created_at=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Chat bot endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing chat request"
        )
        