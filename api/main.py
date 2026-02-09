from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.config.settings import settings
from api.routes.chatbot_router import chatbot_router
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

app = FastAPI(
    title="FAQs Chatbot for CVMS Website",
    description="Stateless chatbot to answer FAQs efficiently and autonomously",
    version="0.1"
)

# CORS Configuration - Allow request from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.DEV_ORIGIN, settings.PROD_ORIGIN],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],  # Allows all headers
  )

# custom exception handler for rate limit
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    """
    Return proper error message based on api endpoint
    
    :param request: use to get router path
    :type request: Request
    :param exc: handles 429 too many request error
    :type exc: RateLimitExceeded
    """
    # get api endpoint
    endpoint = request.url.path
    message = ""
    if "/api/chat-ai/chat" in endpoint:
        message = "You’re sending requests too quickly. Please wait for a minute."
    elif "/api/chat-ai/health-check" in endpoint:
        message = "Health check requests limit exceeded. Please wait for a minute."
    else:
        message = "Rate limit exceeded. Please slow down."
        
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate Limit Exceeded",
            "message": message,
            "endpoint": endpoint
        }
    )
    
        
app.include_router(chatbot_router)

# API Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Chatbot API",
        "endpoints": {
            "health_check": "/api/chat-ai/health-check",
            "chatbot_route": "/api/chat-ai/chat"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    