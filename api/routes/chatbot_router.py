from fastapi import APIRouter, Depends, HTTPException, Header, Request, status

from slowapi import Limiter
from slowapi.util import get_remote_address

from datetime import datetime, timezone
import asyncio
import logging

from api.config.settings import settings
from api.scripts.chatbot import retriever
from api.schemas.chatbot_schemas import *
from api.services.chatbot_service import chatbot_service

logger = logging.getLogger(__name__)

chatbot_router = APIRouter(prefix="/api/chat-ai", tags=["chatbot"])
limiter = Limiter(key_func=get_remote_address)


@chatbot_router.get("/health-check")
@limiter.limit("15/minute")
async def health_check(request: Request):
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


async def verify_request_key(request_secret_key: str = Header(None)):
    """Verify the request secret key"""
    if request_secret_key != settings.REQUEST_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid request"
        )


@chatbot_router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    _: None = Depends(verify_request_key)  # hash secret key dependency
):
    """
    Chat endpoint - returns complete response from chatbot
    """
    try:
        # if filled, the request likely made by bot 
        if chat_request.honeypot:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request"
            )
        
        ai_response = await chatbot_service.get_chat_response(chat_request.message)
        
        return ChatResponse(
            role="assistant",
            message=ai_response,
            created_at=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing chat request"
        )
        