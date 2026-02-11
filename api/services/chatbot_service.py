import asyncio
import logging
from typing import Optional
import redis

from api.config.settings import settings
from api.scripts.chatbot import chatbot

logger = logging.getLogger(__name__)


class ChatbotService:
    """Service layer for chatbot business logic"""
    
    def __init__(self):
        self.max_attempts: int = 3
        self.retry_delay: float = 1.0
        self.redis_client = redis.from_url(settings.get_redis_client_uri())
    
    
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
                
                # Validate response
                if self._is_valid_response(ai_response):
                    logger.info(f"Successfully got response on attempt {attempt}")
                    
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
    
    
    @staticmethod
    def _normalize_cache_key(message: str) -> str:
        """
        Normalize message for consistent caching
        
        Removes punctuation, converts to lowercase
        """
        # Remove trailing punctuation and convert to lowercase
        normalized = message.lower().strip().rstrip('?!.,;:')
        return f"faq:{normalized}"  # Prefix for organization
    
    
    @staticmethod
    def _is_valid_response(response: Optional[str]) -> bool:
        """Check if response is valid and non-empty"""
        return bool(response and response.strip())
    
    
    @staticmethod
    def _get_empty_message_response() -> str:
        """Fallback response for empty messages"""
        return (
            "It looks like the question didn't come through. "
            "Could you please provide the question you'd like answered?"
        )

chatbot_service = ChatbotService()
