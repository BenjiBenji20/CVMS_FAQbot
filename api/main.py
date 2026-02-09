from fastapi import FastAPI
from api.config.settings import settings
from api.routes.chatbot_router import chatbot_router
from fastapi.middleware.cors import CORSMiddleware

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
    