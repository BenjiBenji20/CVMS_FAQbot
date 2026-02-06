from fastapi import FastAPI
from api.routes.chatbot_router import chatbot_router

app = FastAPI(
    title="FAQs Chatbot for CVMS Website",
    description="Stateless chatbot to answer FAQs efficiently and autonomously",
    version="0.1"
)

app.include_router(chatbot_router)

# API Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Chatbot API",
        "endpoints": {
            "health_check": "/api/chatbot/",
            "test_connection": "/api/chatbot/test-connection"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    