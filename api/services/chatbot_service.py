import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
import logging
from typing import Optional
from fastapi import WebSocket
import redis
import uuid

from api.config.settings import settings
from api.scripts.chatbot import chatbot, llm_message_rephraser

logger = logging.getLogger(__name__)


class ChatbotService:
    """Service layer for chatbot business logic"""
    
    def __init__(self):
        self.max_attempts: int = 3
        self.retry_delay: float = 1.0
        # redis client (upstash)
        self.redis_client = redis.from_url(
            settings.get_redis_client_uri(),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # Rate limiting tracking
        self.message_counts: dict[str, list[datetime]] = defaultdict(list)
        self.rate_limit_window = 60  # seconds
        self.rate_limit_max = 10  # messages per window
        self.active_connections: dict[str, WebSocket] = {}
    
    
    async def check_rate_limit(self, session_id: str) -> tuple[bool, int, int]:
        """
        Check rate limit using Redis
        
        Returns:
            (is_allowed, remaining_quota, retry_after_seconds)
        """
        rate_limit_key = f"ws_rate_limit:{session_id}"
        
        try:
            # Get current count from Redis
            current_count = self.redis_client.get(rate_limit_key)
            current_count = int(current_count) if current_count else 0
            
            # Check if limit exceeded
            if current_count >= self.rate_limit_max:
                ttl = self.redis_client.ttl(rate_limit_key)
                retry_after = ttl if ttl > 0 else self.rate_limit_window
                logger.warning(f"Rate limit exceeded for session {session_id}")
                return False, 0, retry_after
            
            # Increment counter and set/refresh expiry
            pipe = self.redis_client.pipeline()
            pipe.incr(rate_limit_key)
            pipe.expire(rate_limit_key, self.rate_limit_window)
            pipe.execute()
            
            remaining = self.rate_limit_max - current_count - 1
            return True, remaining, 0
            
        except redis.RedisError as e:
            logger.error(f"Redis rate limit check failed: {e}. Allowing request.")
            # Fail open - allow request if Redis is down
            return True, self.rate_limit_max, 0
        
    
    async def get_rate_limit_info(self, session_id: str) -> dict:
        """
        Get current rate limit status for a session
        """
        rate_limit_key = f"ws_rate_limit:{session_id}"
        
        try:
            current_count = self.redis_client.get(rate_limit_key)
            current_count = int(current_count) if current_count else 0
            ttl = self.redis_client.ttl(rate_limit_key)
            
            return {
                "limit": self.rate_limit_max,
                "remaining": max(0, self.rate_limit_max - current_count),
                "reset_in": ttl if ttl > 0 else self.rate_limit_window,
                "window": self.rate_limit_window
            }
        except redis.RedisError as e:
            logger.error(f"Failed to get rate limit info: {e}")
            return {
                "limit": self.rate_limit_max,
                "remaining": self.rate_limit_max,
                "reset_in": self.rate_limit_window,
                "window": self.rate_limit_window
            }
    
    
    async def connect_ws(self, ws: WebSocket) -> str:
        """accept connection and return temporary session id"""
        await ws.accept()
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = ws
        logger.info(f"Client connected: {session_id}")
        return session_id

    
    async def disconnect_ws(self, session_id: str):
        """Unregister websocket connection and cleanup Redis data"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"Client disconnected: {session_id}")
        
        # Cleanup rate limit data from Redis
        try:
            rate_limit_key = f"ws_rate_limit:{session_id}"
            self.redis_client.delete(rate_limit_key)
            logger.info(f"Cleaned up Redis data for session: {session_id}")
        except redis.RedisError as e:
            logger.warning(f"Failed to cleanup Redis data: {e}")
            
            
    async def send_to_client(self, session_id: str, data: dict):
        """Send message to specific client"""
        ws = self.active_connections.get(session_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")

    
    async def get_chat_response(self, message: str) -> str:
        """
        Get response from chatbot with retry logic
        
        Args:
            message: User's question (already validated and stripped)
            
        Returns:
            str: AI response
            
        Raises:
            Exception: If all retry attempts fail
        """
        # Handle edge case: empty message after strip
        if not message or message.isspace():
            return self._get_empty_message_response()
        
        # Normalize message for cache key (case-insensitive, no punctuation)
        cache_key = self._normalize_cache_key(message)
        
        # checks if faqs as key value pair (quest and answer pair) exists in redis db
        # if exists, return it immediately, not let embeddings and augmentation process inside the script
        try:
            cached_response = self.redis_client.get(cache_key)
            if cached_response:
                return cached_response 
        except redis.RedisError as e:
            logger.warning(f"Redis error during cache check: {e}. Proceeding without cache.")
        
        # Retry logic
        ai_response = ""
        last_error = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                # Call chatbot function in thread (it's blocking)
                ai_response = await asyncio.to_thread(chatbot, message)
                
                # If fallback message exists in initial ai_response, rephrase
                CORE_FALLBACK = "cvmscustomerservice@gmail.com"
                if CORE_FALLBACK in ai_response.lower() and not message.startswith("REPHRASED:"):
                    logger.info("LLM couldn't answer. Rephrasing query...")
                    rephrased_message = await asyncio.to_thread(llm_message_rephraser, message)
                    
                    # request again with to_rephrase = True signaling that this is a second 
                    # semantic search with rephrased message
                    ai_response = await asyncio.to_thread(
                            chatbot, f"REPHRASE: {rephrased_message}", True
                        )

                # Validate response
                if self._is_valid_response(ai_response):
                    logger.info(f"Successfully got response on attempt {attempt}")
                    
                    # Don't cache fallback
                    if CORE_FALLBACK not in ai_response.lower():
                        # store new response as value and message is the key
                        try:
                            self.redis_client.setex(
                                name=cache_key,
                                time=172800,  # 2 days expiration
                                value=ai_response
                            )
                            logger.info(f"Cached response for: {message}")
                        except redis.RedisError as e:
                            logger.warning(f"Redis error during cache set: {e}")
                            
                    return ai_response
                
                logger.warning(f"Empty response on attempt {attempt}/{self.max_attempts}")
                
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt}/{self.max_attempts} failed: {str(e)}")
                
                # Don't retry on last attempt
                if attempt < self.max_attempts:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception(
                        f"Failed to get response after {self.max_attempts} attempts"
                    ) from last_error
        
        # If all attempts returned empty responses
        raise Exception(
            f"Chatbot returned empty responses after {self.max_attempts} attempts"
        )
    
    
    def _normalize_cache_key(self, message: str) -> str:
        """
        Normalize message for consistent caching
        Removes punctuation, converts to lowercase
        """
        # Remove trailing punctuation and convert to lowercase
        normalized = message.lower().strip().rstrip('?!.,;:')
        return f"faq:{normalized}"  # Prefix for organization
    
    
    def _is_valid_response(self, response: Optional[str]) -> bool:
        """Check if response is valid and non-empty"""
        return bool(response and response.strip())
    
    
    def _get_empty_message_response(self) -> str:
        """Fallback response for empty messages"""
        return (
            "It looks like the question didn't come through. "
            "Could you please provide the question you'd like answered?"
        )

chatbot_service = ChatbotService()
