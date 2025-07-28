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
# Set environment variables to use transformers cache
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/torch/sentence_transformers

# Download the model and its dependencies
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-small-en'); print('Model downloaded successfully')"

# Copy application code
COPY . .

# Copy and make the wrapper script executable
COPY run_all.sh .
RUN chmod +x run_all.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PDF_FOLDER=/app/collections1
# Set offline mode by default
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Expose port (if needed for web interface)
EXPOSE 8000

# Use the wrapper script as entrypoint
ENTRYPOINT ["./run_all.sh"]
CMD ["collections1"]

