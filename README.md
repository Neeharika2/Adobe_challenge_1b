# PDF Document Processor

A comprehensive system that processes PDF documents to generate structured input and output files for document analysis tasks. The system uses semantic search with the all-MiniLM-L6-v2 model to find relevant content based on user-defined personas and tasks.

## Features

- **PDF Text Extraction**: Extracts text from PDF files page by page
- **Text Chunking**: Intelligently splits text into manageable chunks for processing
- **Semantic Search**: Uses all-MiniLM-L6-v2 embeddings for semantic similarity matching
- **Keyword Matching**: Combines keyword search with semantic similarity
- **Flexible Input Generation**: Creates input.json files with customizable persona and tasks
- **Structured Output**: Generates output.json with ranked relevant sections and analysis

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (first run only):
```python
import nltk
nltk.download('punkt')
```

## File Structure

```
adobe-1b/
├── pdf_processor.py          # Core processing logic
├── interactive_processor.py  # Interactive and batch processing modes
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── samplecase/
│   ├── pdfs/                # Place your PDF files here
│   ├── input.json           # Sample input format
│   └── output.json          # Sample output format
└── output/                  # Generated files will be saved here
```

## Usage

### Interactive Mode

Run the interactive processor to input your persona and job requirements:

```bash
python interactive_processor.py
```

Choose option 1 for interactive mode and follow the prompts:
- **Challenge ID**: Unique identifier for your task
- **Test Case Name**: Name for the test case
- **Description**: Brief description of the task
- **Persona Role**: The role (e.g., "Travel Planner", "Research Analyst")
- **Job to be done**: Detailed description of what needs to be accomplished
- **Output Directory**: Where to save the generated files

### Batch Mode

For automated processing, use batch mode with a configuration file:

```bash
python interactive_processor.py
```

Choose option 2. This will create a `config.json` file if it doesn't exist. Edit the config file and run again.

Sample `config.json`:
```json
{
    "challenge_id": "batch_001",
    "test_case_name": "batch_analysis", 
    "description": "Batch Document Analysis",
    "persona_role": "Document Analyst",
    "job_task": "Analyze and extract key information from all provided documents",
    "output_dir": "batch_output"
}
```

### Direct Usage

You can also use the PDFProcessor class directly in your code:

```python
from pdf_processor import PDFProcessor

processor = PDFProcessor(pdf_folder="path/to/pdfs")

input_data, output_data = processor.process_documents(
    challenge_id="custom_001",
    test_case_name="analysis",
    description="Custom Analysis",
    persona_role="Analyst",
    job_task="Extract relevant information for decision making",
    output_dir="results"
)
```

## How It Works

1. **PDF Loading**: The system scans the PDF folder and extracts text from each PDF file page by page.

2. **Text Chunking**: Large text blocks are split into smaller chunks (default 500 characters) for better processing.

3. **Embedding Generation**: Each text chunk is converted to embeddings using the all-MiniLM-L6-v2 model.

4. **Keyword Extraction**: The system extracts relevant keywords from the job description.

5. **Relevance Scoring**: Each text chunk is scored based on:
   - Keyword matches (30% weight)
   - Semantic similarity to the task (70% weight)

6. **Section Ranking**: The most relevant sections are ranked and selected for the output.

7. **Output Generation**: The system generates structured JSON files with:
   - Input configuration
   - Top relevant sections with metadata
   - Summarized and refined text content

## Input Format

The generated `input.json` follows this structure:

```json
{
    "challenge_info": {
        "challenge_id": "example_001",
        "test_case_name": "document_analysis",
        "description": "Document Analysis Task"
    },
    "documents": [
        {
            "filename": "document1.pdf",
            "title": "Document 1 Title"
        }
    ],
    "persona": {
        "role": "Document Analyst"
    },
    "job_to_be_done": {
        "task": "Analyze and extract relevant information"
    }
}
```

## Output Format

The generated `output.json` includes:

```json
{
    "metadata": {
        "input_documents": ["doc1.pdf", "doc2.pdf"],
        "persona": "Document Analyst",
        "job_to_be_done": "Task description",
        "processing_timestamp": "2025-07-23T10:30:00"
    },
    "extracted_sections": [
        {
            "document": "doc1.pdf",
            "section_title": "Relevant Section Title",
            "importance_rank": 1,
            "page_number": 5
        }
    ],
    "subsection_analysis": [
        {
            "document": "doc1.pdf", 
            "refined_text": "Summarized and refined content...",
            "page_number": 5
        }
    ]
}
```

## Customization

### Chunk Size
Modify the chunk size in the PDFProcessor initialization:
```python
processor = PDFProcessor()
processor.chunk_size = 800  # Increase chunk size
```

### Relevance Scoring
Adjust the weights for keyword vs semantic similarity in the `find_relevant_sections` method:
```python
combined_score = (keyword_score * 0.4) + (similarity * 0.6)  # More weight to keywords
```

### Number of Results
Change the number of returned sections:
```python
relevant_sections = self.find_relevant_sections(documents, keywords, task, top_k=10)
```

## Troubleshooting

1. **PDF extraction issues**: Some PDFs may have text as images. Consider using OCR tools like Tesseract for such files.

2. **Memory issues**: For large PDF collections, process documents in batches or increase chunk size.

3. **Poor relevance**: Adjust the keyword extraction logic or similarity weights based on your specific use case.

4. **Model loading**: The first run may take time to download the sentence transformer model.

## Dependencies

- `PyPDF2`: PDF text extraction
- `sentence-transformers`: Semantic embeddings with all-MiniLM-L6-v2
- `scikit-learn`: Cosine similarity calculations
- `nltk`: Text tokenization
- `numpy`: Numerical operations

# Adobe Challenge 1b

## Model Information
This project uses **Sentence-BERT (all-MiniLM-L6-v2)** for generating document embeddings and semantic search functionality. The model provides efficient sentence-level embeddings for similarity calculations.

## Setup Instructions

### Prerequisites
- Python 3.8+
- MongoDB (local or cloud instance)
- Required Python packages (see requirements.txt)

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
**Important:** Before running the application, update the MongoDB collection name in the configuration files according to your preference.

1. Open the database configuration file
2. Locate the `COLLECTION_NAME` variable
3. Change it to your preferred collection name:
   ```python
   COLLECTION_NAME = "your_preferred_collection_name"
   ```

### Running the Application
```bash
python main.py
```

## Features
- Document embedding using Sentence-BERT
- Semantic search capabilities
- MongoDB integration for document storage
- Efficient similarity matching

## Model Details
- **Model:** all-MiniLM-L6-v2
- **Framework:** Sentence-Transformers
- **Embedding Dimension:** 384
- **Use Case:** Semantic similarity and document retrieval