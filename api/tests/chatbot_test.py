from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    """Test chatbot health check endpoint"""
    response = client.get("/api/chat-ai/health-check")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "unhealthy"]


def test_chat_endpoint_valid_message():
    """Test chat endpoint with valid message"""
    response = client.post(
        "/api/chat-ai/chat",
        json={"message": "What is a decoder?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    assert len(data["message"]) > 0
    assert "created_at" in data


def test_chat_endpoint_empty_message():
    """Test chat endpoint with empty message"""
    # this will return a synthetic valid response
    response = client.post(
        "/api/chat-ai/chat",
        json={"message": ""}
    )
    assert response.status_code == 422  # Validation error


def test_chat_endpoint_whitespace_only():
    """Test chat endpoint with whitespace-only message"""
    # this will return a synthetic valid response
    response = client.post(
        "/api/chat-ai/chat",
        json={"message": "   "}
    )
    assert response.status_code == 422


def test_chatbot_function():
    from scripts.chatbot import chatbot
    
    response = chatbot("test query")
    assert isinstance(response, str)
    assert len(response) > 0
