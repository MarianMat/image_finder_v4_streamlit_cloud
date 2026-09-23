import re
from pathlib import Path
from uuid import uuid4


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        name,
    ) or "image"


def process_image(
    uploaded_file,
    images_dir: Path,
    store,
):
    content = uploaded_file.getvalue()

    if not content:
        raise ValueError("Plik jest pusty.")

    extension = Path(
        uploaded_file.name
    ).suffix.lower()

    if extension not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:
        raise ValueError(
            f"Nieobsługiwany format: {extension}"
        )

    image_id = str(uuid4())

    image_path = (
        images_dir
        / f"{image_id}_{_safe_filename(uploaded_file.name)}"
    )

    image_path.write_bytes(content)

    try:
        description = store.ai.describe_image(
            image_path
        )

        store.add_image(
            image_id=image_id,
            image_path=image_path,
            filename=uploaded_file.name,
            description=description,
        )

        return {
            "image_id": image_id,
            "filename": uploaded_file.name,
            "image_path": str(image_path),
            "description": description,
        }

    except Exception:
        if image_path.exists():
            image_path.unlink()
        raise
