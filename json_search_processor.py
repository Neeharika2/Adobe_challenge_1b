import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings("ignore", message="Failed to find CUDA.")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class JSONSearchProcessor:
    def __init__(self, json_folder: str = None):
        self.json_folder = json_folder or os.environ.get('JSON_FOLDER', 'collections1')
        # Load the model with offline mode support
        try:
            # Set environment variables for offline mode
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            # Try to load from cache (offline mode)
            self.model = SentenceTransformer("BAAI/bge-small-en")
        except Exception as e:
            print(f"Failed to load cached model in offline mode: {e}")
            try:
                # Clear offline environment variables and try online
                if 'HF_HUB_OFFLINE' in os.environ:
                    del os.environ['HF_HUB_OFFLINE']
                if 'TRANSFORMERS_OFFLINE' in os.environ:
                    del os.environ['TRANSFORMERS_OFFLINE']
                    
                # Fallback to online download
                self.model = SentenceTransformer("BAAI/bge-small-en")
            except Exception as e2:
                print(f"Failed to load model online: {e2}")
                raise RuntimeError("Could not load sentence transformer model. Please ensure the model is cached or internet connection is available.")
        
        self.documents = {}
        self.chunk_size = 400
        
        # Initialize pattern cache
        
        self._pattern_cache = {}
        self._task_analysis_cache = {}

    def chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """Enhanced chunking with semantic boundaries and overlap"""
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        # Identify natural breakpoints (headings, numbered items, etc.)
        breakpoint_patterns = [
            r'^\d+\.\s+',  # Numbered lists
            r'^[A-Z][^.!?]*:$',  # Headings ending with colon
            r'^Chapter\s+\d+',  # Chapter headings
            r'^Section\s+\d+',  # Section headings
            r'^\*\*[^*]+\*\*',  # Bold headings in markdown
            r'^#{1,6}\s+',  # Markdown headers
        ]
        
        for i, sentence in enumerate(sentences):
            # Check if this sentence is a natural breakpoint
            is_breakpoint = any(re.match(pattern, sentence.strip()) for pattern in breakpoint_patterns)
            
            # If adding this sentence would exceed chunk size or hit a breakpoint
            if len(current_chunk) + len(sentence) > chunk_size or (is_breakpoint and current_chunk):
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def load_documents(self) -> Dict[str, Dict]:
        """Load and process all JSON documents from json_files folder"""
        documents = {}
        
        # Convert to absolute path if relative
        if not os.path.isabs(self.json_folder):
            self.json_folder = os.path.join(os.getcwd(), self.json_folder)
        
        # Look for json_files subfolder
        json_files_folder = os.path.join(self.json_folder, "json_files")
        
        if not os.path.exists(json_files_folder):
            print(f"JSON files folder not found: {json_files_folder}")
            print(f"Please run create_json_sections.py first to create individual JSON files")
            return documents
        
        # Get all JSON files from json_files folder
        try:
            json_files = []
            all_files = os.listdir(json_files_folder)
            
            for item in all_files:
                if item.endswith('.json'):
                    json_files.append(item)
            
            if not json_files:
                print(f"No JSON files found in {json_files_folder}")
                return documents
                
        except Exception as e:
            print(f"Error reading folder {json_files_folder}: {e}")
            return documents
            
        for filename in json_files:
            json_path = os.path.join(json_files_folder, filename)
            
            try:
                file_size = os.path.getsize(json_path)
                if file_size == 0:
                    print(f"Skipping {filename}: file is empty")
                    continue
                    
                print(f"Processing {filename} ({file_size/1024:.1f} KB)...")
                
                # Load the individual JSON file created by create_json_sections.py
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                if not json_data or 'sections' not in json_data:
                    print(f"Warning: No sections found in {filename}")
                    continue

                # Extract sections data
                sections = json_data['sections']
                document_info = json_data.get('document_info', {})
                original_filename = document_info.get('filename', filename.replace('.json', '.pdf'))
                
                if not sections:
                    print(f"Warning: No sections found in {filename}")
                    continue

                document_data = {
                    'filename': original_filename,
                    'title': document_info.get('title', filename.replace('.json', '')),
                    'sections': [],
                    'chunks': [],
                    'embeddings': []
                }
                
                # Process each section
                for section in sections:
                    title = section.get('title', 'Untitled Section')
                    context = section.get('context', '')
                    page = section.get('page_number', 1)
                    section_type = section.get('section_type', 'section')
                    
                    # Store section with title and context
                    document_data['sections'].append({
                        'title': title,
                        'context': context,
                        'page': page,
                        'section_type': section_type
                    })
                    
                    # Create chunks from the context for search
                    if context and len(context.strip()) > 10:
                        chunks = self.chunk_text(context)
                        
                        for chunk in chunks:
                            document_data['chunks'].append({
                                'text': chunk,
                                'title': title,  # Store section title with chunk
                                'page': page,
                                'section_type': section_type,
                                'embedding': None
                            })
                
                # Generate embeddings for all chunks
                chunk_texts = [chunk['text'] for chunk in document_data['chunks']]
                if chunk_texts:
                    embeddings = self.model.encode(chunk_texts)
                    for i, embedding in enumerate(embeddings):
                        document_data['chunks'][i]['embedding'] = embedding
                
                documents[original_filename] = document_data
                print(f"Successfully processed {filename}: {len(sections)} sections, {len(document_data['chunks'])} chunks")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
                
        return documents
    
    def generate_input_json(self, documents: Dict[str, Dict], 
                          challenge_id: str = "generated_001",
                          test_case_name: str = "document_analysis",
                          description: str = "Document Analysis Task",
                          persona_role: str = "Document Analyst",
                          job_task: str = "Analyze and extract relevant information from documents") -> Dict:
        """Generate input.json format from loaded documents"""
        
        input_data = {
            "challenge_info": {
                "challenge_id": challenge_id,
                "test_case_name": test_case_name,
                "description": description
            },
            "documents": [],
            "persona": {
                "role": persona_role
            },
            "job_to_be_done": {
                "task": job_task
            }
        }
        
        for filename, doc_data in documents.items():
            input_data["documents"].append({
                "filename": filename,
                "title": doc_data['title']
            })
            
        return input_data

    def extract_keywords_from_task(self, task: str) -> List[str]:
        """Extract keywords from task"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]+\b', task.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:15]  # Return top 15 keywords

    def find_relevant_sections(self, documents: Dict[str, Dict], 
                             keywords: List[str], 
                             task: str, 
                             top_k: int = 10) -> List[Dict]:
        """Find relevant sections based on keywords and semantic similarity"""
        
        # Encode task
        task_embedding = self.model.encode([task])
        
        relevant_sections = []
        
        for filename, doc_data in documents.items():
            for chunk_data in doc_data['chunks']:
                chunk_text = chunk_data['text']
                chunk_title = chunk_data.get('title', 'Untitled Section')
                page_num = chunk_data['page']
                section_type = chunk_data.get('section_type', 'section')
                chunk_embedding = chunk_data['embedding']
                
                # Calculate semantic similarity
                semantic_score = 0
                if chunk_embedding is not None:
                    semantic_score = float(cosine_similarity(task_embedding, chunk_embedding.reshape(1, -1))[0][0])
                
                # Calculate keyword matching score
                keyword_score = 0
                chunk_lower = chunk_text.lower()
                title_lower = chunk_title.lower()
                
                for keyword in keywords:
                    # Check both chunk text and title
                    if keyword.lower() in chunk_lower:
                        keyword_score += 1
                    if keyword.lower() in title_lower:
                        keyword_score += 0.5  # Title matches get half weight
                
                # Normalize keyword score
                keyword_score = keyword_score / max(len(keywords), 1)
                
                # Combined score
                combined_score = semantic_score * 0.7 + keyword_score * 0.3
                
                if combined_score > 0.1:  # Threshold
                    relevant_sections.append({
                        'document': filename,
                        'text': chunk_text,
                        'section_title': chunk_title,  # Use section_title instead of title
                        'page_number': page_num,
                        'section_type': section_type,
                        'combined_score': combined_score
                    })
        
        # Sort by score and return top k
        relevant_sections.sort(key=lambda x: x['combined_score'], reverse=True)
        return relevant_sections[:top_k]

    def generate_output_json(self, input_data: Dict, 
                           documents: Dict[str, Dict], 
                           relevant_sections: List[Dict]) -> Dict:
        """Generate output.json format based on relevant sections found"""
        
        output_data = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in input_data["documents"]],
                "persona": input_data["persona"]["role"],
                "job_to_be_done": input_data["job_to_be_done"]["task"],
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_analyzed": sum(len(doc['chunks']) for doc in documents.values()),
                "relevant_sections_found": len(relevant_sections)
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        # Deduplicate sections based on document + section_title combination
        seen_sections = set()
        unique_sections = []
        
        for section in relevant_sections:
            section_key = (section['document'], section['section_title'])
            if section_key not in seen_sections:
                seen_sections.add(section_key)
                unique_sections.append(section)
        
        # Create extracted_sections using unique sections only
        for i, section in enumerate(unique_sections[:5]):  # Top 5 unique sections
            section_title = section['section_title']
            output_data["extracted_sections"].append({
                "document": section['document'],
                "section_title": section_title,
                "importance_rank": i + 1,
                "page_number": section['page_number']
            })
        
        # Create subsection_analysis using unique sections
        for section in unique_sections[:5]:
            # Summarize text to 400 characters max
            refined_text = section['text']
            if len(refined_text) > 400:
                refined_text = refined_text[:397] + "..."
            
            output_data["subsection_analysis"].append({
                "document": section['document'],
                "section_title": section['section_title'],
                "refined_text": refined_text,
                "page_number": section['page_number']
            })
        
        return output_data
    
    def process_documents(self, 
                         challenge_id: str = "generated_001",
                         test_case_name: str = "document_analysis", 
                         description: str = "Document Analysis Task",
                         persona_role: str = "Document Analyst",
                         job_task: str = "Analyze and extract relevant information from documents"
                        ) -> Tuple[Dict, Dict]:
        """Main processing function"""
        
        print("Loading JSON documents...")
        documents = self.load_documents()
        
        if not documents:
            print("No documents found!")
            return None, None
        
        print(f"Loaded {len(documents)} documents")
        
        # Generate input.json
        input_data = self.generate_input_json(
            documents, challenge_id, test_case_name, description, persona_role, job_task
        )
        
        # Extract keywords from task
        keywords = self.extract_keywords_from_task(job_task)
        print(f"Extracted keywords: {keywords}")
        
        # Find relevant sections
        print("Finding relevant sections...")
        relevant_sections = self.find_relevant_sections(documents, keywords, job_task)
        
        # Generate output.json
        output_data = self.generate_output_json(input_data, documents, relevant_sections)
        
        # Save files in the same directory as the JSON folder
        input_path = os.path.join(self.json_folder, "input.json")
        output_path = os.path.join(self.json_folder, "output.json")
        
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(input_data, f, indent=4, ensure_ascii=False)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"Generated {input_path}")
        print(f"Generated {output_path}")
        
        return input_data, output_data

def main():
    # Read existing input.json to get persona and job info
    import sys
    
    collection_folder = sys.argv[1] if len(sys.argv) > 1 else 'collections1'
    
    processor = JSONSearchProcessor(collection_folder)
    
    # Get user inputs
    print("Enter challenge configuration:")
    challenge_id = input("Challenge ID (default: generated_001): ").strip() or "generated_001"
    test_case_name = input("Test case name (default: document_analysis): ").strip() or "document_analysis"
    description = input("Description (default: Document Analysis Task): ").strip() or "Document Analysis Task"
    
    print("\nEnter persona details:")
    persona_role = input("Persona role (default: Document Analyst): ").strip() or "Document Analyst"
    
    print("\nEnter job to be done:")
    job_task = input("Job task (default: Analyze and extract relevant information from documents): ").strip() or "Analyze and extract relevant information from documents"
    
    # Try to read existing input.json for fallback values if user provides empty inputs
    input_path = os.path.join(collection_folder, "input.json")
    if os.path.exists(input_path):
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                existing_input = json.load(f)
            
            # Use existing values as fallbacks if user didn't provide new ones
            if challenge_id == "generated_001":
                challenge_id = existing_input.get('challenge_info', {}).get('challenge_id', challenge_id)
            if test_case_name == "document_analysis":
                test_case_name = existing_input.get('challenge_info', {}).get('test_case_name', test_case_name)
            if description == "Document Analysis Task":
                description = existing_input.get('challenge_info', {}).get('description', description)
            if persona_role == "Document Analyst":
                persona_role = existing_input.get('persona', {}).get('role', persona_role)
            if job_task == "Analyze and extract relevant information from documents":
                job_task = existing_input.get('job_to_be_done', {}).get('task', job_task)
                
        except Exception as e:
            print(f"Warning: Could not read existing input.json: {e}")
    
    print(f"\nProcessing with:")
    print(f"Challenge ID: {challenge_id}")
    print(f"Test case name: {test_case_name}")
    print(f"Description: {description}")
    print(f"Persona role: {persona_role}")
    print(f"Job task: {job_task}")
    
    processor.process_documents(
        challenge_id=challenge_id,
        test_case_name=test_case_name,
        description=description,
        persona_role=persona_role,
        job_task=job_task
    )

if __name__ == "__main__":
    main()