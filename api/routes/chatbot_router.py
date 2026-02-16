from fastapi import APIRouter, Depends, HTTPException, Header, Request, WebSocket, WebSocketDisconnect, status

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
        
        
@chatbot_router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with rate limiting
    
    Client sends:
    {
        "message": "What are your services?",
        "message_id": "optional-tracking-id",
        "website": ""  // honeypot
    }
    
    Server responds:
    {
        "type": "connected|typing|response|error|rate_limit",
        "message": "...",
        "created_at": "...",
        "session_id": "...",
        "message_id": "...",
        "rate_limit": {
            "remaining": 9,
            "limit": 10,
            "reset_in": 60
        }
    }
    """
    # Connect and get session ID
    session_id = await chatbot_service.connect_ws(websocket)
    
    try:
        # Get initial rate limit info
        rate_limit_info = await chatbot_service.get_rate_limit_info(session_id)
        
        # Send welcome message
        await chatbot_service.send_to_client(session_id, {
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to CVMS FAQ Bot 👋",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rate_limit": rate_limit_info
        })
        
        # Listen for messages
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message = data.get("message", "").strip()
            message_id = data.get("message_id")
            honeypot = data.get("website", "")
            
            # ===== RATE LIMIT CHECK =====
            is_allowed, remaining, retry_after = await chatbot_service.check_rate_limit(session_id)
            
            if not is_allowed:
                logger.warning(f"[{session_id}] Rate limit exceeded")
                await chatbot_service.send_to_client(session_id, {
                    "type": "rate_limit",
                    "error": "Rate limit exceeded. Please slow down.",
                    "message": f"You've sent too many messages. Please wait {retry_after} seconds.",
                    "message_id": message_id,
                    "rate_limit": {
                        "remaining": 0,
                        "limit": chatbot_service.rate_limit_max,
                        "reset_in": retry_after
                    }
                })
                continue  # Don't process this message
            
            # ===== HONEYPOT CHECK =====
            if honeypot:
                logger.warning(f"[{session_id}] Honeypot triggered")
                await chatbot_service.send_to_client(session_id, {
                    "type": "error",
                    "error": "Invalid request",
                    "message_id": message_id
                })
                continue
            
            # ===== EMPTY MESSAGE CHECK =====
            if not message:
                await chatbot_service.send_to_client(session_id, {
                    "type": "error",
                    "error": "Empty message",
                    "message": "Please enter a message.",
                    "message_id": message_id
                })
                continue
            
            logger.info(f"[{session_id}] ({remaining} remaining) Message: {message[:50]}...")
            
            # ===== SEND TYPING INDICATOR =====
            await chatbot_service.send_to_client(session_id, {
                "type": "typing",
                "message_id": message_id,
                "rate_limit": {
                    "remaining": remaining,
                    "limit": chatbot_service.rate_limit_max,
                    "reset_in": chatbot_service.rate_limit_window
                }
            })
            
            try:
                # ===== GET AI RESPONSE =====
                ai_response = await chatbot_service.get_chat_response(message)
                
                # ===== SEND RESPONSE =====
                await chatbot_service.send_to_client(session_id, {
                    "type": "response",
                    "role": "assistant",
                    "message": ai_response,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "message_id": message_id,
                    "rate_limit": {
                        "remaining": remaining - 1,
                        "limit": chatbot_service.rate_limit_max,
                        "reset_in": chatbot_service.rate_limit_window
                    }
                })
                
            except Exception as e:
                logger.exception(f"[{session_id}] Error processing message: {e}")
                await chatbot_service.send_to_client(session_id, {
                    "type": "error",
                    "error": "Processing failed",
                    "message": "Sorry, I encountered an error. Please try again.",
                    "message_id": message_id
                })
    
    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected normally")
        await chatbot_service.disconnect_ws(session_id)
    
    except Exception as e:
        logger.exception(f"[{session_id}] WebSocket error: {e}")
        await chatbot_service.disconnect_ws(session_id)
