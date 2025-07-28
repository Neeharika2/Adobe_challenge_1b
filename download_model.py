import os
from sentence_transformers import SentenceTransformer

def download_model():
    """Download the sentence transformer model for offline use"""
    print("Downloading BAAI/bge-small-en model...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download and save the model
    model = SentenceTransformer("BAAI/bge-small-en")
    model.save("models/BAAI/bge-small-en")
    
    print("Model downloaded and saved to models/BAAI/bge-small-en")
    print("You can now build your Docker image with: docker build -t pdf-processor .")

if __name__ == "__main__":
    download_model()
