import os
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace
from unstructured.documents.elements import Title, NarrativeText, ListItem, Table, Image
from dotenv import load_dotenv
from database_manager import store_elements_in_databases
from vision_handler import describe_image

load_dotenv() # Load environment variables from .env file

def process_and_ingest_pdf(file_path):
    print(f"Processing PDF: {file_path}")
    
    # Partition the PDF to extract elements (text, tables, images)
    # Using 'by_title' strategy to try and respect document structure
    # extract_images_in_pdf ensures images are found, infer_table_structure for tables
    elements = partition_pdf(
        filename=file_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=1500, # Max characters per chunk
        new_after_n_chars=1200 # Creates a new chunk after this many chars if current is too long
    )

    processed_elements = []
    for i, el in enumerate(elements):
        cleaned_text = clean_extra_whitespace(el.text)
        
        # Add a unique ID to each element for tracking
        el.metadata.element_id = f"{os.path.basename(file_path)}_{i}"

        if isinstance(el, Table):
            # For tables, store the HTML representation for rich querying later
            el.metadata.text_as_html = getattr(el.metadata, 'text_as_html', cleaned_text)
            processed_elements.append({"type": "Table", "content": el.metadata.text_as_html, "metadata": el.metadata})
            print(f"  - Extracted Table: {cleaned_text[:50]}...")
        elif isinstance(el, Image):
            # If the image has a path, describe it using the vision model
            if el.metadata.image_path:
                description = describe_image(el.metadata.image_path)
                processed_elements.append({"type": "Image", "content": description, "metadata": el.metadata})
                print(f"  - Described Image: {description[:50]}...")
            else:
                # If image path is not available, try to use any existing text
                processed_elements.append({"type": "Image", "content": cleaned_text, "metadata": el.metadata})
        else: # Treat other elements (Title, NarrativeText, ListItem) as general text
            processed_elements.append({"type": "Text", "content": cleaned_text, "metadata": el.metadata})
            print(f"  - Extracted Text: {cleaned_text[:50]}...")
            
    # Store the processed elements in our databases
    store_elements_in_databases(processed_elements)
    print(f"Finished processing and ingesting {file_path}")

if __name__ == "__main__":
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: '{data_dir}' directory not found. Please create it and place your PDFs inside.")
        exit()

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{data_dir}'. Please add some PDFs to process.")
    
    for pdf_file in pdf_files:
        full_path = os.path.join(data_dir, pdf_file)
        process_and_ingest_pdf(full_path)
    
    print("\nAll specified PDFs processed and ingested!")
