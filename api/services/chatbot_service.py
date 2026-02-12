import asyncio
import logging
from typing import Optional
import redis

from api.config.settings import settings
from api.scripts.chatbot import chatbot, llm_message_rephraser

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
        
        # Handle greetings early (skip retrieval)
        if self._is_greeting(message):
            return "Hello! 👋 How can I assist you with information about CVMS today?"
        
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
                ai_response, scores = await asyncio.to_thread(chatbot, message)

                # If no relevant docs found, rephrase and retry
                if len(scores) < 1:
                    rephrased_message = await asyncio.to_thread(llm_message_rephraser, message)
                    
                    # request again with to_rephrase = True signaling that this is a second 
                    # semantic search with rephrased message
                    ai_response, _ = await asyncio.to_thread(chatbot, rephrased_message, True)

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
    
    
    def _normalize_cache_key(self, message: str) -> str:
        """
        Normalize message for consistent caching
        Removes punctuation, converts to lowercase
        """
        # Remove trailing punctuation and convert to lowercase
        normalized = message.lower().strip().rstrip('?!.,;:')
        return f"faq:{normalized}"  # Prefix for organization
    
    
    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good day', 'good afternoon', 'good evening']
        message_lower = message.lower().strip()
        return any(greeting in message_lower for greeting in greetings)
    
    
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
