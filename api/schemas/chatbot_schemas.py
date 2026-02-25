from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import re

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=500,
        description="User's message"
    )
    qa_id: Optional[str] = Field(default=None, description="Direct QA entry mapping")
    action_id: Optional[str] = Field(default=None, description="Direct action entry mapping")

    honeypot: str = Field(default="", alias="website")

    @field_validator('message')
    def validate_message(cls, v: str):
        v = v.strip()
        # Calculate special character ratio
        special_char_ratio = sum(not c.isalnum() for c in v) / len(v)
        # Collapse repeated punctuation
        v = re.sub(r'([?!.,;:])\1+', r'\1', v)

        # remove trailing punctuation if too noisy
        if special_char_ratio > 0.5:
            v = v.rstrip('?!.,;:')

        return v
    

class ActionLink(BaseModel):
    id: str
    title: str
    url: str
    button_text: str
    
    
class MessageSuggestion(BaseModel):
    text: str
    qa_id: Optional[str] = None
    action_id: Optional[str] = None


class ChatResponse(BaseModel):
    role: str
    message: str
    created_at: datetime
    actions: List[ActionLink] = []
    message_suggestions: List[MessageSuggestion] = []
    
    
class ChatReactRequest(BaseModel):
    user_query: str
    is_like: bool

    
class ChatReactResponse(BaseModel):
    action: str
    likes: int
    dislikes: int
    cache_deleted: bool
    reason: str = None
    