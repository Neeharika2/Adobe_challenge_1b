# Technical Approach

## Model Selection
We have implemented **Sentence-BERT (all-MiniLM-L6-v2)** as our core embedding model for the following reasons:

### Why Sentence-BERT?
- **Efficiency:** Lightweight model suitable for real-time applications
- **Quality:** Provides high-quality sentence-level embeddings
- **Speed:** Fast inference time compared to larger transformer models
- **Versatility:** Works well across various text similarity tasks

### Model Specifications
- **Model Name:** all-MiniLM-L6-v2
- **Architecture:** Based on BERT with sentence pooling
- **Embedding Dimension:** 384
- **Max Sequence Length:** 256 tokens
- **Framework:** sentence-transformers library

## Implementation Approach
1. **Document Processing:** Text preprocessing and chunking
2. **Embedding Generation:** Using Sentence-BERT to create vector representations
3. **Storage:** MongoDB for document and embedding storage
4. **Retrieval:** Cosine similarity for semantic search

## Configuration Note
**Important:** Users must update the MongoDB collection name in the configuration before running the application. This allows for personalized database organization and prevents conflicts in shared environments.

## Performance Considerations
- The model balances accuracy and speed effectively
- Suitable for production environments with moderate to high query volumes
- Memory efficient with 384-dimensional embeddings
