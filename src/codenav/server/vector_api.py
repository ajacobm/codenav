"""
Vector API Routes

FastAPI routes for embedding operations using LiteLLM.
"""

import logging
import os
from typing import List, Union, Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default embedding model if not specified in environment or request
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    input: Union[str, List[str]] = Field(
        ..., 
        description="Text to embed - can be a single string or list of strings"
    )
    model: Optional[str] = Field(
        None,
        description="Optional model override (defaults to LITELLM_EMBEDDING_MODEL env var)"
    )


class EmbeddingObject(BaseModel):
    """Individual embedding result."""
    object: str = Field("embedding", description="Type of object")
    index: int = Field(..., description="Index in the input list")
    embedding: List[float] = Field(..., description="Embedding vector")


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    total_tokens: int = Field(..., description="Total number of tokens")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    object: str = Field("list", description="Type of object")
    data: List[EmbeddingObject] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used for embeddings")
    usage: UsageInfo = Field(..., description="Token usage information")


def create_vector_api_router() -> APIRouter:
    """Create FastAPI router with vector/embedding endpoints."""
    
    router = APIRouter(prefix="/api/vector", tags=["vector"])

    @router.post("/embedding", response_model=EmbeddingResponse)
    async def create_embedding(request: EmbeddingRequest) -> Dict[str, Any]:
        """
        Generate embeddings for input text using LiteLLM.
        
        Supports multiple embedding providers through LiteLLM:
        - Azure OpenAI (via AZURE_* env vars)
        - OpenAI (via OPENAI_API_KEY)
        - Ollama (local deployments)
        - And many more through LiteLLM proxy
        
        Configuration via environment variables:
        - LITELLM_EMBEDDING_MODEL: Default model to use (e.g., "text-embedding-3-small")
        - AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION: For Azure OpenAI
        - OPENAI_API_KEY: For OpenAI
        """
        try:
            # Import litellm here to make it an optional dependency
            try:
                from litellm import embedding
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="LiteLLM not installed. Install with: pip install 'codenav[web]'"
                )
            
            # Determine model to use
            model = request.model or os.environ.get("LITELLM_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            
            # Normalize input to list
            texts = request.input if isinstance(request.input, list) else [request.input]
            
            if not texts:
                raise HTTPException(status_code=400, detail="Input cannot be empty")
            
            logger.info(f"Generating embeddings for {len(texts)} text(s) using model: {model}")
            
            # Call LiteLLM embedding function
            # LiteLLM handles routing to the appropriate provider based on model name
            # and environment variables
            response = embedding(
                model=model,
                input=texts
            )
            
            # LiteLLM returns an object with data, model, and usage
            # Extract embeddings from response
            embeddings_data = []
            for idx, embedding_obj in enumerate(response.data):
                # Handle both dict and object-style access for LiteLLM response
                embedding_vector = (
                    embedding_obj.get("embedding") if isinstance(embedding_obj, dict)
                    else getattr(embedding_obj, "embedding", None)
                )
                if embedding_vector is None:
                    raise ValueError(f"Invalid embedding format at index {idx}")
                
                embeddings_data.append(
                    EmbeddingObject(
                        object="embedding",
                        index=idx,
                        embedding=embedding_vector
                    )
                )
            
            # Extract usage information
            usage = response.usage
            usage_info = UsageInfo(
                prompt_tokens=usage.prompt_tokens,
                total_tokens=usage.total_tokens
            )
            
            # Build response
            result = EmbeddingResponse(
                object="list",
                data=embeddings_data,
                model=response.model,
                usage=usage_info
            )
            
            logger.info(f"Successfully generated {len(embeddings_data)} embeddings")
            return result.model_dump()
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embeddings: {str(e)}"
            )
    
    return router
