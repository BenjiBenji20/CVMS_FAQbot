from datetime import datetime
from pydantic import BaseModel, Field, field_validator

import re
from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=500,
        description="User's message"
    )

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

    

class ChatResponse(BaseModel):
    role: str
    message: str
    created_at: datetime