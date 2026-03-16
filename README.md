# FAQBot: Context-Aware Chatbot 🤖

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=database&logoColor=white)](https://www.trychroma.com/)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=ai&logoColor=white)](https://groq.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

Context-aware AI chatbot for public customer service that automates FAQs using document retrieval, guided buttons, and follow-up suggestions using Retrieval Augmented Generation (RAG) architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Vector Store](#running-the-vector-store)
  - [Starting the API Server](#starting-the-api-server)
  - [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)

---

## 🎯 Overview

FAQBot is an intelligent chatbot system designed to automate customer service for the website by providing **24/7 instant, context-aware responses** to frequently asked questions. Built on a stateless architecture, the chatbot eliminates the need for user authentication while delivering accurate answers based on your documentation.

The system leverages **Retrieval-Augmented Generation (RAG)** to combine the power of large language models with your specific knowledge base, ensuring responses are grounded in your actual documentation rather than hallucinated information. This approach significantly reduces customer service workload while maintaining high-quality, accurate responses at any time of day.

---

## ✨ Features

- **🔍 Context-Aware Responses** - Retrieves relevant information from your documentation before answering
- **📚 Document-Based Knowledge** - Answers are grounded in your PDF documentation
- **⚡ Fast Response Time** - Powered by Groq's high-performance LLM infrastructure
- **🔄 Automatic Retry Logic** - Ensures reliable responses with 3-attempt retry mechanism
- **🏥 Health Check Endpoint** - Monitor system status and vector store connectivity
- **🔒 Stateless Architecture** - No authentication required, instant access
- **📊 Semantic Search** - Uses vector embeddings for intelligent document retrieval
- **🎨 RESTful API** - Easy integration with any frontend or application

---

## 🏗️ Architecture

<p align="center">
  <img width="1548" height="845" alt="Image" src="https://github.com/user-attachments/assets/bfb23775-f872-4edd-80f0-652c829b71ba" />
</p>
```

**Flow:**
1. User sends a question via API
2. Vector store retrieves top 5 relevant document chunks
3. Chatbot builds prompt with retrieved context
4. Groq LLM generates response based on context
5. API returns formatted response with timestamp

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend Framework** | FastAPI + Uvicorn | High-performance async API server |
| **Vector Database** | ChromaDB | Persistent storage for document embeddings |
| **Embeddings** | Google Gemini Embedding 001 | Convert text to semantic vectors |
| **LLM** | Groq (Llama 3.3 70B) | Fast inference for response generation |
| **Orchestration** | LangChain | Document processing and retrieval pipeline |
| **PDF Processing** | PyPDF | Extract text from documentation |
| **Language** | Python 3.10+ | Core application logic |

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Google AI Studio API key
- Groq API key

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/BenjiBenji20/CVMS_FAQbot.git
   cd faqbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create required directories**
   ```bash
   mkdir -p api/documents api/chroma_db
   ```

5. **Add your PDF documents**
   ```bash
   # Place your FAQ documents in api/documents/
   cp your-faq-docs.pdf api/documents/
   ```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Google AI Studio API Key (for embeddings)
EMBEDDING_MODEL_API_KEY=your_google_api_key_here
MODEL_NAME=models/gemini-embedding-001

# Groq API Key (for LLM)
LLM_API_KEY=your_groq_api_key_here
LLM_NAME=openai/gpt-oss-120b
```

### Get API Keys

- **Google AI Studio**: https://ai.google.dev/
- **Groq**: https://console.groq.com/keys

---

## 🚀 Usage

### Running the Vector Store

**First-time setup** (index your documents):

```bash
python -m api.scripts.vector_store
```

This will:
- Load all PDFs from `api/documents/`
- Split documents into chunks
- Generate embeddings
- Store in `api/chroma_db/`

> **Note**: Re-run this command whenever you add new documents.

### Starting the API Server

**Development mode:**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Production mode:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at: `http://localhost:8000`

### API Endpoints

#### 📡 Health Check
```bash
GET /api/chat-ai/health-check
```

**Response:**
```json
{
  "status": "healthy",
  "vector_store": "connected",
  "documents_in_store": true
}
```

#### 💬 Chat
```bash
POST /api/chat-ai/chat
Content-Type: application/json

{
  "message": "What are your business hours?"
}
```

**Response:**
```json
{
  "role": "assistant",
  "message": "Based on the information provided, our business hours are Monday to Friday, 9 AM to 5 PM.",
  "created_at": "2026-02-06T10:30:00Z"
}
```

### Testing with cURL

```bash
# Health check
curl http://localhost:8000/api/chat-ai/health-check

# Send a question
curl -X POST http://localhost:8000/api/chat-ai/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I reset my password?"}'
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation.

---

## 📁 Project Structure

```
faqbot/
├── api/
│   ├── config/
│   │   └── settings.py          # Configuration management
│   ├── routers/
│   │   └── chatbot_router.py    # API endpoints
│   ├── scripts/
│   │   ├── vector_store.py      # Document indexing
│   │   └── chatbot.py           # Chatbot logic
│   ├── documents/               # PDF files (gitignored)
│   ├── chroma_db/               # Vector embeddings (gitignored)
│   └── main.py                  # FastAPI application
│   ├── tests/
│   │   └── chatbot_test.py      # Unit tests
├── .env                         # Environment variables (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📝NOTE
In api/scripts/vector_store.py line 35 specifically for variable doc_text_splitter adjust the configuration based on whats needed to your document chunking.


## 🧪 Testing

### Run Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest api/tests/ -v

# Run specific test
pytest api/tests/test_chatbot.py::test_health_check -v
```

### Manual Testing

**Using Postman:**
1. Import the API collection from `/docs` endpoint
2. Test health check endpoint
3. Test chat endpoint with sample questions

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/chat-ai/chat",
    json={"message": "What are the available products?"}
)
print(response.json())
```

---

## 🔑 Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `EMBEDDING_MODEL_API_KEY` | Google AI Studio API key for embeddings | `AIza...` |
| `MODEL_NAME` | Google embedding model name | `models/gemini-embedding-001` |
| `LLM_API_KEY` | Groq API key for LLM | `gsk_...` |
| `LLM_NAME` | Groq model name | `openai/gpt-oss-120b` |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Write unit tests for new features
- Follow PEP 8 style guide
- Update documentation for API changes
- Add docstrings to functions

## 🙏 Acknowledgments

- **FastAPI** - Modern web framework
- **ChromaDB** - Vector database
- **LangChain** - LLM orchestration
- **Groq** - High-performance LLM inference
- **Google AI** - Embedding models

---

## 📞 Support

For issues and questions:
- 🐛 [Report a bug](https://github.com/BenjiBenji20/CVMS_FAQbot.git/issues)
- 💡 [Request a feature](https://github.com/BenjiBenji20/CVMS_FAQbot.git/issues)
- 📧 Email: benjicanones6@gmail.com

---
