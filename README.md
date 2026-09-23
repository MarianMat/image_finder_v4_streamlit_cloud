# AI Image Finder V4 — Streamlit Cloud

Wersja przygotowana bezpośrednio pod deployment na Streamlit Community Cloud.

## Pipeline

Zdjęcie
→ zapis
→ GPT-4o opis
→ text-embedding-3-large
→ Qdrant Cloud
→ embedding zapytania
→ wyszukiwanie semantyczne
→ pasujące zdjęcia

## Streamlit Secrets

W Streamlit Cloud → Manage app → Settings → Secrets wpisz:

```toml
OPENAI_API_KEY = "..."
QDRANT_URL = "https://..."
QDRANT_API_KEY = "..."
OPENAI_VISION_MODEL = "gpt-4o"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
QDRANT_COLLECTION = "image_descriptions"
```

## Uruchomienie lokalne

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Ważna uwaga o zdjęciach

W tej wersji Qdrant przechowuje embeddingi i metadata, natomiast pliki zdjęć są zapisywane w `data/images`.

Na Streamlit Community Cloud lokalny filesystem aplikacji nie jest trwałym storage'em. Do produkcyjnej wersji należy przenieść zdjęcia do object storage, np. S3/R2/Supabase Storage, a w Qdrant przechowywać URL zdjęcia.
