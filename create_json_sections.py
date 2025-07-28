import os
import json
import re
import fitz
from datetime import datetime
from unidecode import unidecode
import pandas as pd
import numpy as np

def extract_spans_from_page(page):
    """Extract text spans with properties from a PDF page using PyMuPDF."""
    file_dict = page.get_text("dict")
    blocks = file_dict["blocks"]
    spans = []
    
    for block in blocks:
        if block["type"] == 0:  # Text blocks only
            for line in block["lines"]:
                for span in line["spans"]:
                    text = unidecode(span["text"]).strip()
                    if text.replace(" ", "") != "":
                        spans.append({
                            "xmin": span["bbox"][0],
                            "ymin": span["bbox"][1],
                            "xmax": span["bbox"][2],
                            "ymax": span["bbox"][3],
                            "text": text,
                            "font_size": span["size"],
                            "font": span["font"],
                            "is_bold": "bold" in span["font"].lower(),
                            "is_upper": re.sub(r"[\(\[].*?[\)\]]", "", text).isupper()
                        })
    return spans

def assign_tags(span_df):
    """Assign tags (h, p, s) based on font size and style."""
    span_scores = []
    special = r'[(_:/,#%\=@)]'
    
    for _, row in span_df.iterrows():
        score = round(row["font_size"])
        text = row["text"]
        if not re.search(special, text):
            if row["is_bold"]:
                score += 1
            if row["is_upper"]:
                score += 1
        span_scores.append(score)

    values, counts = np.unique(span_scores, return_counts=True)
    style_dict = dict(zip(values, counts))
    p_size = max(style_dict, key=style_dict.get)
    
    tag_dict = {}
    idx = 0
    for size in sorted(values, reverse=True):
        idx += 1
        if size == p_size:
            idx = 0
            tag_dict[size] = "p"
        elif size > p_size:
            tag_dict[size] = f"h{idx}"
        else:
            tag_dict[size] = f"s{idx}"
    
    span_df["tag"] = [tag_dict[score] for score in span_scores]
    return span_df

def extract_section_title_from_text(text: str) -> str:
    """Extract a meaningful section title from text content"""
    if not text or not text.strip():
        return "Untitled Section"
    
    lines = text.strip().split('\n')
    
    # Look for lines that appear to be titles/headings
    for line in lines[:3]:  # Check first 3 lines
        line = line.strip()
        if not line:
            continue
            
        # Skip very short lines (less than 8 characters)
        if len(line) < 8:
            continue
            
        # Handle introduction/conclusion cases
        if line.lower().startswith(('introduction', 'conclusion')):
            # Get the next 4 words after introduction/conclusion
            words = line.split()
            if len(words) > 1:  # Has words after introduction/conclusion
                next_words = words[1:5]  # Take next 1-4 words
                if next_words:
                    return ' '.join(next_words)
            
            # If no meaningful words after introduction/conclusion, look in next lines
            for next_line in lines[1:4]:
                next_line = next_line.strip()
                if len(next_line) > 10:
                    words = re.findall(r'\b[A-Za-z]+\b', next_line)
                    if len(words) >= 2:
                        return ' '.join(words[:4])
            
            # Fallback for introduction/conclusion
            continue
            
        # Skip common non-title words and irrelevant instruction-type titles
        skip_words = {'done', 'apply', 'ok', 'yes', 'no', 'next', 'back', 'continue', 'cancel', 'close', 'open', 'save', 'edit', 'delete', 'remove', 'add', 'copy', 'paste', 'cut', 'undo', 'redo', 'select', 'all', 'none', 'file', 'tools', 'help', 'view', 'window', 'format', 'insert', 'table', 'image', 'text', 'page', 'document', 'preferences', 'settings', 'options', 'menu', 'button', 'click', 'press', 'type', 'enter', 'choose', 'select'}
        irrelevant_phrases = {'instruction','introduction', 'introductions','extract' 'instructions', 'introduction instructions', 'general introduction', 'section introduction', 'chapter introduction', 'overview introduction', 'getting started', 'before you begin', 'preliminary information', 'overview', 'summary', 'conclusion', 'appendix', 'references', 'bibliography', 'glossary', 'note', 'notice', 'warning', 'caution', 'tip', 'hint', 'advice'}
        
        # Check if line is an irrelevant instruction-type title
        if (line.lower().strip() in skip_words or 
            any(phrase in line.lower() for phrase in irrelevant_phrases)):
            continue
            
        # Skip lines that are just punctuation or numbers
        if re.match(r'^[^\w]*$', line) or re.match(r'^\d+\.?$', line):
            continue
            
        # Check if line looks like a meaningful title
        if (line.endswith(':') or 
            (len(line) >= 8 and len(line) < 100 and not line.endswith('.') and 
             not line.lower().startswith(('the ', 'a ', 'an ', 'in ', 'on ', 'at ', 'this ', 'that ', 'to ', 'if ', 'when ', 'where ', 'how ', 'you ', 'it ', 'and ', 'or ', 'but ')) and
             not re.match(r'^\d+\s*$', line)) or  # Not just numbers
            re.match(r'^[A-Z][A-Za-z\s]+[A-Za-z]$', line) or  # Proper case without punctuation
            re.match(r'^\d+\.\s+[A-Za-z]{3,}', line)):  # Numbered headings with substantial text
            return line
    
    # If no good title found, extract first 1-4 meaningful words from paragraph content
    # Find the first substantial paragraph (skip short lines)
    for line in lines:
        line = line.strip()
        if len(line) > 20:  # Find a substantial line with content
            words = re.findall(r'\b[A-Za-z]+\b', line)
            if len(words) >= 1:
                # Take first 1-4 words depending on their length
                title_words = []
                char_count = 0
                for word in words[:4]:
                    if char_count + len(word) > 50:  # Don't exceed 50 chars
                        break
                    title_words.append(word)
                    char_count += len(word) + 1  # +1 for space
                    
                    # Stop if we have enough meaningful content
                    if len(title_words) >= 2 and char_count > 15:
                        break
                
                if title_words:
                    return ' '.join(title_words)
    
    # Fallback: use first sentence or first 80 characters, but make sure it's meaningful
    first_sentence = text.split('.')[0].strip()
    if 15 <= len(first_sentence) <= 120 and not first_sentence.lower().startswith(('the ', 'a ', 'an ', 'in ', 'on ', 'at ', 'this ', 'that ', 'to ', 'if ', 'when ', 'you ', 'it ')):
        return first_sentence
    
    # Look for the first substantial phrase (more than 15 chars)
    for line in lines[:5]:
        line = line.strip()
        if 15 <= len(line) <= 120 and not line.lower().startswith(('the ', 'a ', 'an ', 'in ', 'on ', 'at ', 'this ', 'that ', 'to ', 'if ', 'when ', 'you ', 'it ')):
            return line
    
    # Final fallback
    return text[:80].strip() + ("..." if len(text) > 80 else "")

def process_pdf_to_sections(pdf_path):
    """Process a PDF and extract sections with titles and context"""
    sections = []
    
    try:
        with fitz.open(pdf_path) as doc:
            all_spans = []
            
            # Extract all spans from all pages
            for page_num, page in enumerate(doc, 1):
                spans = extract_spans_from_page(page)
                for span in spans:
                    span["page_number"] = page_num
                all_spans.extend(spans)
            
            if not all_spans:
                print(f"No text found in {pdf_path}")
                return sections
            
            span_df = pd.DataFrame(all_spans)
            span_df = assign_tags(span_df)
            
            # Group content by pages and create sections
            for page_num in range(1, len(doc) + 1):
                page_spans = span_df[span_df["page_number"] == page_num]
                
                if page_spans.empty:
                    continue
                
                # Get headings for this page
                headings = page_spans[page_spans["tag"].str.contains("h", na=False)]
                regular_text = page_spans[~page_spans["tag"].str.contains("h", na=False)]
                
                # If we have headings, create sections based on them, but filter and group intelligently
                if not headings.empty:
                    # Filter headings to only meaningful ones
                    meaningful_headings = []
                    
                    for _, heading_row in headings.iterrows():
                        title = heading_row["text"].strip()
                        
                        # Skip very short headings or common UI elements, and irrelevant instruction titles
                        skip_words = {'done', 'apply', 'ok', 'yes', 'no', 'next', 'back', 'continue', 'cancel', 'close', 'open', 'save', 'edit', 'delete', 'remove', 'add', 'copy', 'paste', 'cut', 'select', 'all', 'none', 'file', 'tools', 'help', 'view', 'window', 'format', 'insert', 'table', 'image', 'text', 'page', 'document', 'preferences', 'settings', 'options', 'menu', 'button'}
                        irrelevant_phrases = {'instruction introduction', 'instructions introduction', 'introduction instructions', 'general introduction', 'section introduction', 'chapter introduction', 'overview introduction', 'getting started', 'before you begin', 'preliminary information'}
                        
                        if (len(title) >= 8 and 
                            title.lower() not in skip_words and
                            not any(phrase in title.lower() for phrase in irrelevant_phrases) and
                            not re.match(r'^\d+\.?$', title) and  # Not just numbers
                            not re.match(r'^[^\w]*$', title)):  # Not just punctuation
                            meaningful_headings.append(heading_row)
                    
                    # Create sections from meaningful headings
                    for i, heading_row in enumerate(meaningful_headings):
                        title = heading_row["text"]
                        
                        # Find content after this heading (until next meaningful heading or end of page)
                        heading_idx = page_spans.index[page_spans.index == heading_row.name][0]
                        
                        # Find next meaningful heading
                        if i + 1 < len(meaningful_headings):
                            next_heading_idx = page_spans.index[page_spans.index == meaningful_headings[i + 1].name][0]
                            content_spans = page_spans[(page_spans.index > heading_idx) & 
                                                      (page_spans.index < next_heading_idx)]
                        else:
                            content_spans = page_spans[page_spans.index > heading_idx]
                        
                        # Combine title and content
                        content_texts = [title] + content_spans["text"].tolist()
                        context = "\n".join(content_texts).strip()
                        
                        # Check if title is still irrelevant even after filtering
                        irrelevant_phrases = {'instructions', 'introduction', 'instructions introduction', 'introduction instructions', 'general introduction', 'section introduction', 'chapter introduction', 'overview introduction', 'ingredients'}
                        if any(phrase in title.lower() for phrase in irrelevant_phrases):
                            # Extract better title from content
                            content_only = "\n".join(content_spans["text"].tolist()).strip()
                            if content_only:
                                title = extract_section_title_from_text(content_only)
                        
                        # Only create section if there's substantial content
                        if len(context) > 20:
                            sections.append({
                                "title": title,
                                "context": context,
                                "page_number": page_num,
                                "section_type": "heading_section"
                            })
                else:
                    # No headings, treat entire page as one section
                    page_text = "\n".join(page_spans["text"].tolist())
                    title = extract_section_title_from_text(page_text)
                    
                    sections.append({
                        "title": title,
                        "context": page_text,
                        "page_number": page_num,
                        "section_type": "page_content"
                    })
            
            # Post-process sections to combine those without meaningful titles
            sections = combine_sections_without_titles(sections)
                    
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
    
    return sections

def has_meaningful_title(title: str) -> bool:
    """Check if a title is meaningful or should be combined with previous section"""
    if not title or len(title.strip()) < 8:
        return False
    
    # Check for generic/meaningless titles
    meaningless_patterns = [
        r'^Untitled Section$',
        r'^Page \d+$',
        r'^Section \d+$',
        r'^Chapter \d+$',
        r'^\d+$',
        r'^[^\w]*$',  # Only punctuation
        r'^The [a-z\s]{1,15}$',  # Very short "The something"
        r'^A [a-z\s]{1,15}$',    # Very short "A something"
        r'^An [a-z\s]{1,15}$',   # Very short "An something"
    ]
    
    for pattern in meaningless_patterns:
        if re.match(pattern, title.strip(), re.IGNORECASE):
            return False
    
    # Check for generic words that indicate no real title
    generic_words = {'content', 'information', 'data', 'details', 'text', 'section', 'part', 'chapter', 'page'}
    title_words = set(title.lower().split())
    
    # If title is only generic words, it's not meaningful
    if title_words.issubset(generic_words):
        return False
    
    return True

def combine_sections_without_titles(sections: list) -> list:
    """Combine sections that don't have meaningful titles with the previous section"""
    if not sections:
        return sections
    
    combined_sections = []
    
    for i, section in enumerate(sections):
        title = section.get('title', '')
        context = section.get('context', '')
        
        # Check if section starts with introduction or conclusion
        context_lower = context.lower().strip()
        is_intro_conclusion = (context_lower.startswith('introduction') or 
                             context_lower.startswith('conclusion') or
                             context_lower.startswith('ingredients') or
                             any(line.strip().lower().startswith(('introduction', 'conclusion', 'ingredients')) 
                                 for line in context.split('\n')[:3]))
        
        if is_intro_conclusion:
            # Try to extract title from words after introduction/conclusion
            lines = context.strip().split('\n')
            new_title = None
            
            for line in lines[:3]:
                line = line.strip()
                if line.lower().startswith(('introduction', 'conclusion', 'ingredients')):
                    words = line.split()
                    if len(words) > 1:  # Has words after introduction/conclusion
                        next_words = words[1:5]  # Take next 1-4 words
                        if next_words and len(' '.join(next_words)) > 3:
                            new_title = ' '.join(next_words)
                            break
            
            # If we couldn't extract a good title and there's a previous section, use its title
            if not new_title and combined_sections:
                prev_title = combined_sections[-1]['title']
                # Only reuse title if it's meaningful and not generic
                if (not prev_title.lower().startswith(('section', 'page', 'chapter', 'untitled')) and
                    len(prev_title) > 8):
                    new_title = prev_title
            
            # If we found a new title, use it
            if new_title:
                section['title'] = new_title
                title = new_title
        
        # If this section doesn't have a meaningful title and there's a previous section
        if not has_meaningful_title(title) and combined_sections:
            # Combine with the previous section
            prev_section = combined_sections[-1]
            
            # Combine contexts
            combined_context = prev_section['context'] + "\n\n" + section['context']
            
            # Update the previous section
            combined_sections[-1] = {
                'title': prev_section['title'],
                'context': combined_context,
                'page_number': prev_section['page_number'],  # Keep original page number
                'section_type': prev_section['section_type']
            }
            
            print(f"Combined section '{title}' with previous section '{prev_section['title']}'")
        else:
            # Add as new section (but improve title if it's still not great)
            if not has_meaningful_title(title):
                # Try to extract a better title from the content
                improved_title = extract_section_title_from_text(section['context'])
                if has_meaningful_title(improved_title):
                    section['title'] = improved_title
                else:
                    # Last resort: use first few words
                    words = re.findall(r'\b[A-Za-z]+\b', section['context'])
                    if len(words) >= 2:
                        section['title'] = ' '.join(words[:3])
                    else:
                        section['title'] = f"Section {i+1}"
            
            combined_sections.append(section)
    
    return combined_sections

def create_json_files_for_collection(collection_path):
    """Create individual JSON files for each PDF in a collection"""
    pdfs_folder = os.path.join(collection_path, "pdfs")
    json_files_folder = os.path.join(collection_path, "json_files")
    
    if not os.path.exists(pdfs_folder):
        print(f"PDFs folder not found: {pdfs_folder}")
        return
    
    # Create json_files folder
    os.makedirs(json_files_folder, exist_ok=True)
    
    # Process each PDF
    for filename in os.listdir(pdfs_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdfs_folder, filename)
            print(f"Processing {filename}...")
            
            sections = process_pdf_to_sections(pdf_path)
            
            if sections:
                # Create JSON structure
                json_data = {
                    "document_info": {
                        "filename": filename,
                        "title": filename.replace('.pdf', ''),
                        "extraction_timestamp": datetime.now().isoformat(),
                        "total_sections": len(sections)
                    },
                    "sections": sections
                }
                
                # Save JSON file
                json_filename = filename.replace('.pdf', '.json')
                json_path = os.path.join(json_files_folder, json_filename)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                print(f"Created: {json_filename} with {len(sections)} sections")

def main():
    import sys
    
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
        # Process the specified collection
        collection_path = os.path.join(os.getcwd(), collection_name)
        if os.path.exists(collection_path):
            print(f"Processing {collection_name}...")
            create_json_files_for_collection(collection_path)
        else:
            print(f"Collection folder '{collection_name}' not found.")
            return
    else:
        # Automatically find all collections* folders in the current working directory
        base_dir = os.getcwd()
        collections_folders = [
            name for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name)) and name.startswith("collections")
        ]
        
        if not collections_folders:
            print("No collections folders found.")
            return

        print("Available collections:")
        for idx, folder in enumerate(collections_folders, 1):
            print(f"{idx}. {folder}")

        collection_name = input("Enter the collection name to process (or leave blank to process all): ").strip()

        if collection_name and collection_name in collections_folders:
            folders_to_process = [collection_name]
        elif collection_name:
            print(f"Collection '{collection_name}' not found. Exiting.")
            return
        else:
            folders_to_process = collections_folders

        for folder in folders_to_process:
            folder_path = os.path.join(base_dir, folder)
            print(f"\nProcessing {folder}...")
            create_json_files_for_collection(folder_path)

if __name__ == "__main__":
    main()