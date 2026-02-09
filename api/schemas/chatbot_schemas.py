from datetime import datetime
from pydantic import BaseModel, Field, field_validator

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1, 
        max_length=500,  # Limit message size
        description="User's message"
    )
    
    honeypot: str = Field(default="", alias="website")  # bots fill this
    
    @field_validator('message')
    def validate_message(cls, v: str):
        # Block excessive special characters
        special_char_ratio = sum(not c.isalnum() for c in v) / len(v)
        if special_char_ratio > 0.3:
            raise ValueError('Too many special characters')
        
        return v.strip()
    

class ChatResponse(BaseModel):
    role: str
    message: str
    created_at: datetime