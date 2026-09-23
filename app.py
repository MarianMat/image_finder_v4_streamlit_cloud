import os
from pathlib import Path

import streamlit as st

from image_service import process_image
from vector_store import ImageVectorStore

st.set_page_config(
    page_title="AI Image Finder V4",
    page_icon="🖼️",
    layout="wide",
)

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read Streamlit Cloud Secrets first, then local environment variables."""
    try:
        value = st.secrets.get(name)
        if value is not None:
            return str(value)
    except Exception:
        pass

    return os.getenv(name, default)


@st.cache_resource
def get_store():
    return ImageVectorStore(
        qdrant_url=get_setting("QDRANT_URL"),
        qdrant_api_key=get_setting("QDRANT_API_KEY"),
        collection_name=get_setting(
            "QDRANT_COLLECTION",
            "image_descriptions",
        ),
    )


try:
    store = get_store()
except Exception as exc:
    st.error("Nie udało się połączyć z Qdrant.")
    st.exception(exc)
    st.stop()


st.title("🖼️ AI Image Finder — V4")
st.caption(
    "Upload zdjęć → opis AI → embedding → Qdrant → "
    "wyszukiwanie semantyczne"
)

with st.sidebar:
    st.header("Ustawienia wyszukiwania")

    top_k = st.slider(
        "Liczba wyników",
        min_value=1,
        max_value=20,
        value=6,
    )

    min_score = st.slider(
        "Minimalne podobieństwo",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )

    st.divider()
    st.caption(
        f"Vision: `{get_setting('OPENAI_VISION_MODEL', 'gpt-4o')}`"
    )
    st.caption(
        "Embedding: "
        f"`{get_setting('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')}`"
    )
    st.caption(
        f"Qdrant collection: "
        f"`{get_setting('QDRANT_COLLECTION', 'image_descriptions')}`"
    )


upload_tab, search_tab = st.tabs(
    ["📤 Dodaj zdjęcia", "🔎 Wyszukaj zdjęcia"]
)


with upload_tab:
    st.subheader("Prześlij jedno lub wiele zdjęć")

    uploaded_files = st.file_uploader(
        "Wybierz zdjęcia",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button(
        "🚀 Przetwórz zdjęcia",
        type="primary",
    ):
        progress = st.progress(0)
        success = 0
        errors = 0

        for index, uploaded_file in enumerate(
            uploaded_files,
            start=1,
        ):
            try:
                result = process_image(
                    uploaded_file=uploaded_file,
                    images_dir=IMAGES_DIR,
                    store=store,
                )

                success += 1

                with st.expander(
                    f"✅ {result['filename']}",
                    expanded=True,
                ):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.image(
                            result["image_path"],
                            use_container_width=True,
                        )

                    with col2:
                        st.markdown("**Opis AI:**")
                        st.write(result["description"])

            except Exception as exc:
                errors += 1
                st.error(
                    f"❌ {uploaded_file.name}: {exc}"
                )

            progress.progress(
                index / len(uploaded_files)
            )

        st.success(
            f"Zakończono. Sukces: {success}, błędy: {errors}."
        )


with search_tab:
    st.subheader("Wyszukiwanie semantyczne")

    query = st.text_input(
        "Czego szukasz?",
        placeholder=(
            "np. czerwony samochód stojący przed budynkiem"
        ),
    )

    if st.button(
        "🔎 Szukaj",
        type="primary",
        disabled=not query.strip(),
    ):
        with st.spinner(
            "Tworzę embedding zapytania i przeszukuję Qdrant..."
        ):
            results = store.search(
                query=query.strip(),
                limit=top_k,
                score_threshold=min_score,
            )

        if not results:
            st.warning(
                "Nie znaleziono zdjęć powyżej ustawionego progu."
            )
        else:
            st.success(
                f"Znaleziono {len(results)} wyników."
            )

            columns = st.columns(3)

            for index, result in enumerate(results):
                with columns[index % 3]:
                    st.image(
                        result["image_path"],
                        use_container_width=True,
                    )
                    st.markdown(
                        f"**{result['filename']}**"
                    )
                    st.caption(
                        f"Similarity: {result['score']:.3f}"
                    )
                    st.write(result["description"])
