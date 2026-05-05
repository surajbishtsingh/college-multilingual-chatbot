FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Set cache paths BEFORE downloading
ENV HF_HOME=/app/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# ✅ Now model downloads to /app/.cache (same place runtime will look)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
