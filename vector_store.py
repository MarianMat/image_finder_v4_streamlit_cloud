from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from openai_service import OpenAIService


class ImageVectorStore:
    VECTOR_SIZE = 3072

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str,
    ):
        if not qdrant_url or not qdrant_api_key:
            raise RuntimeError(
                "Brak QDRANT_URL lub QDRANT_API_KEY. "
                "W Streamlit Cloud dodaj je w Settings → Secrets."
            )

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        self.collection_name = collection_name

        import streamlit as st

        api_key = None
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass

        import os
        api_key = api_key or os.getenv(
            "OPENAI_API_KEY"
        )

        self.ai = OpenAIService(
            api_key=api_key,
            vision_model=(
                st.secrets.get(
                    "OPENAI_VISION_MODEL",
                    os.getenv(
                        "OPENAI_VISION_MODEL",
                        "gpt-4o",
                    ),
                )
                if hasattr(st, "secrets")
                else os.getenv(
                    "OPENAI_VISION_MODEL",
                    "gpt-4o",
                )
            ),
            embedding_model=(
                st.secrets.get(
                    "OPENAI_EMBEDDING_MODEL",
                    os.getenv(
                        "OPENAI_EMBEDDING_MODEL",
                        "text-embedding-3-large",
                    ),
                )
                if hasattr(st, "secrets")
                else os.getenv(
                    "OPENAI_EMBEDDING_MODEL",
                    "text-embedding-3-large",
                )
            ),
        )

        self._ensure_collection()

    def _ensure_collection(self):
        existing = {
            item.name
            for item in self.client.get_collections().collections
        }

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def add_image(
        self,
        image_id: str,
        image_path: Path,
        filename: str,
        description: str,
    ):
        embedding = self.ai.create_embedding(
            description
        )

        payload = {
            "image_id": image_id,
            "filename": filename,
            "image_path": str(image_path),
            "description": description,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

    def search(
        self,
        query: str,
        limit: int,
        score_threshold: float,
    ) -> list[dict]:
        query_embedding = self.ai.create_embedding(
            query
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}
            image_path = Path(
                payload.get("image_path", "")
            )

            if not image_path.exists():
                continue

            results.append(
                {
                    "score": float(point.score),
                    "filename": payload.get(
                        "filename",
                        "unknown",
                    ),
                    "image_path": str(image_path),
                    "description": payload.get(
                        "description",
                        "",
                    ),
                    "image_id": payload.get(
                        "image_id"
                    ),
                }
            )

        return results
