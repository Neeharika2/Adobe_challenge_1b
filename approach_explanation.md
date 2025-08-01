# Approach Explanation

## Methodology

Our solution is designed to automate the extraction, structuring, and semantic search of information from PDF documents, making it easier to retrieve relevant content for specific user queries or tasks. The workflow consists of two main stages: PDF section extraction and semantic search processing.

### 1. PDF Section Extraction

We use the `create_json_sections.py` script to process each PDF file. The script leverages PyMuPDF to extract text spans from each page, capturing not only the text but also formatting features such as font size and boldness. These features are used to heuristically tag each span as a heading, paragraph, or subtext. 

To create meaningful sections, the script groups content based on detected headings. If a heading is deemed irrelevant (e.g., "introduction", "conclusion", or generic terms), we attempt to generate a more descriptive title by extracting the next few words from the content or, if necessary, reusing the previous section's title. If a section lacks a meaningful title, it is merged with the previous section to maintain context and avoid fragmentation. The result is a set of structured sections, each with a title, context, page number, and type, which are saved as JSON files for each PDF.

### 2. Semantic Search and Output Generation

The `json_search_processor.py` script loads the structured JSON files and prepares them for semantic search through a sophisticated chunking and embedding process.

#### Text Chunking Strategy

Each section's content undergoes intelligent chunking to optimize both semantic coherence and processing efficiency:

1. **Sentence-Level Splitting**: The content is first split into individual sentences using NLTK's sentence tokenizer to preserve natural language boundaries
2. **Adaptive Chunk Size**: Sentences are then grouped into chunks with a target size of approximately 200-300 tokens, ensuring chunks are neither too small (losing context) nor too large (diluting semantic focus)
3. **Context Preservation**: When splitting longer sections, we maintain overlap between adjacent chunks (typically 50-100 tokens) to preserve contextual continuity
4. **Semantic Boundary Respect**: Chunks are created respecting paragraph boundaries and logical breaks in the text to maintain semantic coherence

#### Embedding Generation Process

**sentence-transformer embeddings are generated using the `BAAI/bge-small-en` model** for each chunk through the following pipeline:

1. **Preprocessing**: Each chunk is cleaned and normalized (removing excessive whitespace, special characters)
2. **Tokenization**: The BGE model's tokenizer converts text chunks into token sequences
3. **Embedding Generation**: Each chunk is passed through the BAAI/bge-small-en model to generate a 384-dimensional dense vector representation
4. **Normalization**: Embeddings are L2-normalized to enable efficient cosine similarity calculations
5. **Storage**: Both the original chunk text and its corresponding embedding vector are stored with metadata (source section, page number, chunk index)

#### Model Selection: BAAI/bge-small-en

We chose the **BAAI/bge-small-en** model for the following reasons:

- **High Performance**: BGE (BAAI General Embedding) models are state-of-the-art embedding models that consistently rank at the top of MTEB (Massive Text Embedding Benchmark) leaderboards
- **Optimized for Retrieval**: Specifically designed and fine-tuned for information retrieval and semantic search tasks, making it ideal for document search applications
- **Efficiency**: The "small" variant provides excellent performance while maintaining reasonable computational requirements
- **Multilingual Support**: Strong performance on English text with good generalization capabilities
- **Production Ready**: Proven effectiveness in real-world search and retrieval scenarios
- **Recent Architecture**: Benefits from latest advances in embedding model design and training techniques

When a user provides a persona and a job-to-be-done (task), the script extracts keywords from the task and computes both keyword and semantic similarity scores between the task and each chunk using the high-quality embeddings generated by the BAAI/bge-small-en model.

Relevant sections are ranked based on a weighted combination of keyword and semantic similarity. The top results are compiled into an `output.json` file, which includes metadata, a list of the most relevant sections (with accurate titles from the original PDFs), and a brief analysis of each extracted subsection.

## Challenges Faced

- **Title Extraction:** Many PDFs contain generic or irrelevant headings. We addressed this by developing logic to extract better titles from the content or merge such sections with previous ones.
- **Section Fragmentation:** Without careful merging, sections without titles could result in fragmented or contextless output. Our merging logic ensures coherent and meaningful sections.
- **Semantic Search Quality:** Balancing keyword matching and semantic similarity required tuning to ensure relevant results, especially for ambiguous or broad user queries.
- **Dockerization:** Ensuring that all dependencies (including NLTK data and system libraries) were available in the Docker container, and that output files were written to the host system, required careful use of Docker volume mounts and environment setup.

## Dockerfile and Execution Instructions

The `Dockerfile` sets up a Python 3.11 environment with all required dependencies:

1. **System Dependencies**: Installs build tools (gcc, g++, cmake) required for compiling Python packages
2. **Python Dependencies**: Installs PyTorch CPU version and all requirements from `requirements.txt`
3. **NLTK Data**: Downloads required punkt tokenizer data during build
4. **Model Pre-download**: Downloads and caches the BAAI/bge-small-en sentence transformer model during build to enable offline operation
5. **Offline Configuration**: Sets environment variables to force offline mode, preventing network calls during execution
6. **Wrapper Script**: Includes `run_all.sh` script that automatically runs both processing steps sequentially

**Build the Docker image:**
```sh
docker build -t pdf-processor .
```

**Run both scripts sequentially (recommended approach):**
```sh
# Process a specific collection (e.g., collections2)
docker run --rm -it -v "${PWD}:/app" pdf-processor collections2
# Replace collection2 with the required collection name
```

The wrapper script automatically:
1. Runs `create_json_sections.py` to extract sections from PDFs and create JSON files
2. If successful, runs `json_search_processor.py` to generate input.json and output.json
3. Provides clear status updates and error handling

**Run individual scripts (if needed):**
```sh
# Run only section extraction
docker run --rm -it -v "${PWD}:/app" --entrypoint python pdf-processor create_json_sections.py collections2

# Run only search processor (requires JSON files to exist)
docker run --rm -it -v "${PWD}:/app" --entrypoint python pdf-processor json_search_processor.py collections2
```

**Important Notes:**
- Always use the `-v "${PWD}:/app"` flag to mount your working directory, ensuring output files are accessible on your host system
- The model is pre-downloaded during Docker build, so no internet connection is required during execution
- The container runs in offline mode by default to ensure consistent behavior across environments
- Replace `collections2` with your target collection folder name as needed

