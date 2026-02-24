import asyncio
import json
import logging
from typing import List, Optional, Tuple
import redis

from api.config.settings import settings
from api.scripts.chatbot import chatbot, llm_message_rephraser
from api.utils.keywords_normalizer import kw_norm

logger = logging.getLogger(__name__)


class ChatbotService:
    """Service layer for chatbot business logic"""
    
    def __init__(self):
        self.max_attempts: int = 3
        self.retry_delay: float = 1.0
        self.redis_client = redis.from_url(settings.get_redis_client_uri())
        self.CACHED_KEY_TTL = 604800  # 7 days in seconds
    
    
    async def get_chat_response(self, message: str) -> Tuple[str, List[dict]]:
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
            return self._get_empty_message_response(), []
        
        # Transform short hands into complete words
        message = kw_norm.normalize_message(message)
        
        # Normalize message for cache key (case-insensitive, no punctuation)
        cache_key = kw_norm.normalize_cache_key(message)
        
        # checks if faqs as key value pair (quest and answer pair) exists in redis db
        # if exists, return it immediately, not let embeddings and augmentation process inside the script
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Cache hit for: {message[:50]}...")
                parsed = json.loads(cached_data)
                return parsed['message'], parsed.get('actions', [])
        except redis.RedisError as e:
            logger.warning(f"Redis error during cache check: {e}. Proceeding without cache.")
        
        # Retry logic
        ai_response = ""
        actions = []
        last_error = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                # Call chatbot function in thread (it's blocking)
                ai_response, actions = await asyncio.to_thread(chatbot, message)
                
                # If fallback message exists in initial ai_response, rephrase
                CORE_FALLBACK = "facebook messenger"
                if CORE_FALLBACK in ai_response.lower() and not message.startswith("REPHRASED:"):
                    logger.info("LLM couldn't answer. Rephrasing query...")
                    rephrased_message = await asyncio.to_thread(llm_message_rephraser, message)
                    
                    # request again with to_rephrase = True signaling that this is a second 
                    # semantic search with rephrased message
                    ai_response, actions = await asyncio.to_thread(
                            chatbot, f"REPHRASE: {rephrased_message}", True
                        )

                # Validate response
                if self._is_valid_response(ai_response):
                    logger.info(f"Successfully got response on attempt {attempt}")
                    
                    # Don't cache fallback
                    if CORE_FALLBACK not in ai_response.lower():
                        # store new response as value and message is the key
                        try:
                            cache_data = json.dumps({
                                'message': ai_response,
                                'actions': actions
                            })
                            self.redis_client.setex(
                                name=cache_key,
                                time=self.CACHED_KEY_TTL,  # 7 days expiration
                                value=cache_data
                            )
                            logger.info(f"Cached response for: {message}")
                        except redis.RedisError as e:
                            logger.warning(f"Redis error during cache set: {e}")
                            
                    return ai_response, actions
                
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
    
    
    async def chat_react(self, user_query: str, is_like: bool = True) -> dict:
        """
        Handle like/dislike reaction for cached responses.
        Deletes cache if dislikes >= 3 OR dislikes > likes.
        
        Args:
            user_query: Original user question
            is_like: True for like, False for dislike
            
        Returns:
            dict with status and counts
            
        Raises:
            ValueError: If cache key not found
        """
        # Normalize to get cache key
        cache_key = kw_norm.normalize_cache_key(user_query)
        
        try:
            # Check if cached response exists
            if not self.redis_client.exists(cache_key):
                logger.warning(f"Cache key not found: {cache_key}")
                raise ValueError(f"No cached response found for this query")
            
            # Use pipeline to batch Redis operations
            pipe = self.redis_client.pipeline()
            
            if is_like:
                # Increment likes
                pipe.incr(f"{cache_key}:likes")
                pipe.expire(f"{cache_key}:likes", self.CACHED_KEY_TTL)
                pipe.get(f"{cache_key}:likes")
                pipe.get(f"{cache_key}:dislikes")
                results = pipe.execute()
                
                likes = int(results[1] or 0)
                dislikes = int(results[2] or 0)
                
                logger.info(f"Like added for: {cache_key} (Likes: {likes}, Dislikes: {dislikes})")
                
                return {
                    "action": "like_added",
                    "likes": likes,
                    "dislikes": dislikes,
                    "cache_deleted": False
                }
            
            else:
                # Increment dislikes
                pipe.incr(f"{cache_key}:dislikes")
                pipe.expire(f"{cache_key}:dislikes", self.CACHED_KEY_TTL)
                pipe.get(f"{cache_key}:likes")
                pipe.get(f"{cache_key}:dislikes")
                results = pipe.execute()
                
                likes = int(results[1] or 0)
                dislikes = int(results[2] or 0)
                
                # Delete cache if:
                # 1. Dislikes >= 3 (absolute threshold)
                # 2. OR dislikes > likes (majority negative)
                should_delete = dislikes >= 3 or (dislikes > likes and dislikes >= 3)
                
                if should_delete:
                    # Delete cache and counters
                    pipe = self.redis_client.pipeline()
                    pipe.delete(cache_key)
                    pipe.delete(f"{cache_key}:likes")
                    pipe.delete(f"{cache_key}:dislikes")
                    pipe.execute()
                    
                    logger.warning(f"Cache deleted for: {cache_key} (Likes: {likes}, Dislikes: {dislikes})")
                    
                    return {
                        "action": "cache_deleted",
                        "reason": "Too many dislikes",
                        "likes": likes,
                        "dislikes": dislikes,
                        "cache_deleted": True
                    }
                
                else:
                    logger.info(f"Dislike added for: {cache_key} (Likes: {likes}, Dislikes: {dislikes})")
                    
                    return {
                        "action": "dislike_added",
                        "likes": likes,
                        "dislikes": dislikes,
                        "cache_deleted": False
                    }
        
        except redis.RedisError as e:
            logger.error(f"Redis error in chat_react: {e}")
            raise Exception("Failed to process reaction due to cache error")
     
    
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
