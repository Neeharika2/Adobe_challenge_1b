# Approach Explanation

## Methodology

Our solution is designed to automate the extraction, structuring, and semantic search of information from PDF documents, making it easier to retrieve relevant content for specific user queries or tasks. The workflow consists of two main stages: PDF section extraction and semantic search processing.

### 1. PDF Section Extraction

We use the `create_json_sections.py` script to process each PDF file. The script leverages PyMuPDF to extract text spans from each page, capturing not only the text but also formatting features such as font size and boldness. These features are used to heuristically tag each span as a heading, paragraph, or subtext. 

To create meaningful sections, the script groups content based on detected headings. If a heading is deemed irrelevant (e.g., "introduction", "conclusion", or generic terms), we attempt to generate a more descriptive title by extracting the next few words from the content or, if necessary, reusing the previous section's title. If a section lacks a meaningful title, it is merged with the previous section to maintain context and avoid fragmentation. The result is a set of structured sections, each with a title, context, page number, and type, which are saved as JSON files for each PDF.

### 2. Semantic Search and Output Generation

The `json_search_processor.py` script loads the structured JSON files and prepares them for semantic search. Each section's content is chunked into manageable pieces, and sentence-transformer embeddings are generated for each chunk. When a user provides a persona and a job-to-be-done (task), the script extracts keywords from the task and computes both keyword and semantic similarity scores between the task and each chunk.

Relevant sections are ranked based on a weighted combination of keyword and semantic similarity. The top results are compiled into an `output.json` file, which includes metadata, a list of the most relevant sections (with accurate titles from the original PDFs), and a brief analysis of each extracted subsection.

## Challenges Faced

- **Title Extraction:** Many PDFs contain generic or irrelevant headings. We addressed this by developing logic to extract better titles from the content or merge such sections with previous ones.
- **Section Fragmentation:** Without careful merging, sections without titles could result in fragmented or contextless output. Our merging logic ensures coherent and meaningful sections.
- **Semantic Search Quality:** Balancing keyword matching and semantic similarity required tuning to ensure relevant results, especially for ambiguous or broad user queries.
- **Dockerization:** Ensuring that all dependencies (including NLTK data and system libraries) were available in the Docker container, and that output files were written to the host system, required careful use of Docker volume mounts and environment setup.

## Dockerfile and Execution Instructions

The `Dockerfile` sets up a Python 3.11 environment, installs all required system and Python dependencies, downloads NLTK data, and copies the application code. The entrypoint is set to allow flexible execution of either the section extraction or search processor scripts.

**Build the Docker image:**
```sh
docker build -t pdf-processor .
```

**Run section extraction (creates JSON files from PDFs):**
```sh
docker run --rm -it -v "${PWD}:/app" pdf-processor create_json_sections.py
```

**Run the search processor (generates input.json and output.json):**
```sh
docker run --rm -it -v "${PWD}:/app" pdf-processor json_search_processor.py collections1
```
Replace `collections1` with your target folder as needed.

**Note:** Always use the `-v` flag to mount your working directory, ensuring output files are accessible on your host system.

