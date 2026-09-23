import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


class OpenAIService:
    def __init__(
        self,
        api_key: str,
        vision_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-large",
    ):
        if not api_key:
            raise RuntimeError(
                "Brak OPENAI_API_KEY."
            )

        self.client = OpenAI(api_key=api_key)
        self.vision_model = vision_model
        self.embedding_model = embedding_model

    def describe_image(
        self,
        image_path: Path,
    ) -> str:
        mime_type, _ = mimetypes.guess_type(
            image_path.name
        )
        mime_type = mime_type or "image/jpeg"

        encoded = base64.b64encode(
            image_path.read_bytes()
        ).decode("utf-8")

        response = self.client.responses.create(
            model=self.vision_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Opisz zdjęcie po polsku. "
                                "Opis ma być rzeczowy i przydatny "
                                "do późniejszego wyszukiwania "
                                "semantycznego. Uwzględnij główne "
                                "obiekty, osoby bez identyfikowania "
                                "ich, czynności, miejsce/scenę, "
                                "kolory i istotne relacje między "
                                "obiektami. Nie zgaduj informacji, "
                                "których nie można wiarygodnie "
                                "wywnioskować z obrazu."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{mime_type};base64,{encoded}"
                            ),
                        },
                    ],
                }
            ],
        )

        description = response.output_text.strip()

        if not description:
            raise RuntimeError(
                "Model nie zwrócił opisu."
            )

        return description

    def create_embedding(
        self,
        text: str,
    ) -> list[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )

        return response.data[0].embedding
