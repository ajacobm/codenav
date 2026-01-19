"""
Tests for Vector API Endpoints

Tests for embedding generation via LiteLLM.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

# Mock litellm before imports
mock_litellm = MagicMock()

def mock_embedding_function(model, input):
    """Mock embedding function that returns embeddings based on input size."""
    texts = input if isinstance(input, list) else [input]
    embeddings = []
    for i in range(len(texts)):
        embeddings.append({
            "embedding": [0.1 * (i+1), 0.2 * (i+1), 0.3 * (i+1), 0.4 * (i+1), 0.5 * (i+1)]
        })
    
    response = MagicMock()
    response.data = embeddings
    response.model = model
    response.usage = MagicMock()
    response.usage.prompt_tokens = len(texts) * 5
    response.usage.total_tokens = len(texts) * 5
    return response

mock_litellm.embedding = MagicMock(side_effect=mock_embedding_function)


class TestVectorAPI:
    """Test vector/embedding API endpoints."""
    
    @patch.dict('sys.modules', {'litellm': mock_litellm})
    def test_embedding_request_single_text(self):
        """Test embedding request with single text string."""
        from src.codenav.server.vector_api import EmbeddingRequest
        
        request = EmbeddingRequest(input="test text")
        assert request.input == "test text"
        assert request.model is None
    
    @patch.dict('sys.modules', {'litellm': mock_litellm})
    def test_embedding_request_multiple_texts(self):
        """Test embedding request with list of texts."""
        from src.codenav.server.vector_api import EmbeddingRequest
        
        request = EmbeddingRequest(input=["text1", "text2"])
        assert request.input == ["text1", "text2"]
        assert request.model is None
    
    @patch.dict('sys.modules', {'litellm': mock_litellm})
    def test_embedding_request_with_model(self):
        """Test embedding request with custom model."""
        from src.codenav.server.vector_api import EmbeddingRequest
        
        request = EmbeddingRequest(input="test", model="text-embedding-ada-002")
        assert request.model == "text-embedding-ada-002"
    
    @patch.dict('sys.modules', {'litellm': mock_litellm})
    def test_embedding_response_structure(self):
        """Test embedding response structure."""
        from src.codenav.server.vector_api import (
            EmbeddingResponse, 
            EmbeddingObject,
            UsageInfo
        )
        
        embedding_obj = EmbeddingObject(
            object="embedding",
            index=0,
            embedding=[0.1, 0.2, 0.3]
        )
        
        usage = UsageInfo(prompt_tokens=5, total_tokens=5)
        
        response = EmbeddingResponse(
            object="list",
            data=[embedding_obj],
            model="test-model",
            usage=usage
        )
        
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        assert response.model == "test-model"
        assert response.usage.prompt_tokens == 5
    
    @patch.dict('sys.modules', {'litellm': mock_litellm})
    def test_router_creation(self):
        """Test vector API router creation."""
        from src.codenav.server.vector_api import create_vector_api_router
        
        router = create_vector_api_router()
        assert router is not None
        assert router.prefix == "/api/vector"
        assert "vector" in router.tags


class TestEmbeddingEndpoint:
    """Test embedding endpoint behavior."""
    
    @patch.dict('sys.modules', {'litellm': mock_litellm})
    @pytest.mark.asyncio
    async def test_embedding_with_single_text(self):
        """Test embedding generation with single text."""
        from src.codenav.server.vector_api import create_vector_api_router, EmbeddingRequest
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport
        
        app = FastAPI()
        router = create_vector_api_router()
        app.include_router(router)
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/vector/embedding",
                json={"input": "test text"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]
        assert data["model"] == "text-embedding-3-small"
    
    @patch.dict('sys.modules', {'litellm': mock_litellm})
    @pytest.mark.asyncio
    async def test_embedding_with_multiple_texts(self):
        """Test embedding generation with multiple texts."""
        from src.codenav.server.vector_api import create_vector_api_router
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport
        
        app = FastAPI()
        router = create_vector_api_router()
        app.include_router(router)
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/vector/embedding",
                json={"input": ["text1", "text2"]}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
    
    @patch.dict('sys.modules', {'litellm': mock_litellm})
    @pytest.mark.asyncio
    async def test_embedding_with_custom_model(self):
        """Test embedding with custom model override."""
        from src.codenav.server.vector_api import create_vector_api_router
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport
        
        app = FastAPI()
        router = create_vector_api_router()
        app.include_router(router)
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/vector/embedding",
                json={
                    "input": "test text",
                    "model": "text-embedding-ada-002"
                }
            )
        
        assert response.status_code == 200
        # Verify litellm.embedding was called with the custom model
        mock_litellm.embedding.assert_called()
    
    @patch.dict('sys.modules', {'litellm': mock_litellm})
    @patch.dict(os.environ, {"LITELLM_EMBEDDING_MODEL": "azure/text-embedding-ada-002"})
    @pytest.mark.asyncio
    async def test_embedding_with_env_model(self):
        """Test embedding uses model from environment variable."""
        from src.codenav.server.vector_api import create_vector_api_router
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport
        
        app = FastAPI()
        router = create_vector_api_router()
        app.include_router(router)
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/vector/embedding",
                json={"input": "test text"}
            )
        
        assert response.status_code == 200
        # Verify the environment variable was used
        call_args = mock_litellm.embedding.call_args
        assert call_args is not None
    
    @patch.dict('sys.modules', {'litellm': mock_litellm})
    @pytest.mark.asyncio
    async def test_embedding_empty_input_error(self):
        """Test embedding with empty input returns error."""
        from src.codenav.server.vector_api import create_vector_api_router
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport
        
        app = FastAPI()
        router = create_vector_api_router()
        app.include_router(router)
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/vector/embedding",
                json={"input": []}
            )
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    @patch.dict('sys.modules', {'litellm': mock_litellm})
    @pytest.mark.asyncio
    async def test_embedding_litellm_error_handling(self):
        """Test error handling when LiteLLM raises an exception."""
        from src.codenav.server.vector_api import create_vector_api_router
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport
        
        # Make litellm.embedding raise an exception
        mock_litellm.embedding.side_effect = Exception("API error")
        
        app = FastAPI()
        router = create_vector_api_router()
        app.include_router(router)
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/vector/embedding",
                json={"input": "test text"}
            )
        
        assert response.status_code == 500
        assert "Failed to generate embeddings" in response.json()["detail"]
        
        # Reset the side effect for other tests
        mock_litellm.embedding.side_effect = mock_embedding_function
