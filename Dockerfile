FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK punkt data
RUN python -m nltk.downloader punkt punkt_tab

# Pre-download sentence-transformers model to avoid repeated downloads at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en')"

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PDF_FOLDER=/app/collections1

# Expose port (if needed for web interface)
EXPOSE 8000

# Allow command override; default runs json_search_processor.py collections1
ENTRYPOINT ["python"]
CMD ["json_search_processor.py", "collections1"]
