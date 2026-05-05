FROM python:3.11-slim
WORKDIR /app

# Set cache FIRST before anything
ENV HF_HOME=/app/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model to /app/.cache
RUN python -c "
import os
os.environ['HF_HOME'] = '/app/.cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/app/.cache'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print('Model cached successfully')
"

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
