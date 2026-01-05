import streamlit as st
import os
from src.database_manager import search_vector_db, search_table_db
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide", page_title="Enterprise Knowledge Search")

st.title("📄 Enterprise Knowledge Search")
st.markdown("Search across your enterprise PDF documents. This tool leverages semantic search for text and keyword search for extracted tables.")

# Sidebar for information or settings
with st.sidebar:
    st.header("About This Tool")
    st.info("""
        This application converts unstructured PDFs into searchable knowledge.
        It uses a Vector Database for text and image descriptions, and a NoSQL Database for structured tables.
        
        **To get started:**
        1.  Place your PDFs in the `data/` folder.
        2.  Run `python src/ingestion.py` from your terminal.
        3.  Then, run `streamlit run app.py` to launch this interface.
        
        *Ensure your OpenAI API Key and MongoDB URI are set in a `.env` file.*
    """)
    st.markdown("---")
    st.write("Developed for Enterprise Knowledge Retrieval")

# Main search interface
st.header("Ask a Question about your Documents")
query = st.text_input("Enter your query here:", placeholder="e.g., What are the Q3 financial results for product X?")

if query:
    st.subheader("Searching...")
    
    # --- Vector DB Search (Text and Image Descriptions) ---
    st.markdown("#### Relevant Text & Image Descriptions")
    vector_results = search_vector_db(query, num_results=5)

    if vector_results and vector_results['documents'][0]:
        for i, doc in enumerate(vector_results['documents'][0]):
            metadata = vector_results['metadatas'][0][i]
            st.expander(f"**Relevance: {100 - vector_results['distances'][0][i]:.2f}%** - From {metadata.get('file_path', 'N/A')} (Page {metadata.get('page_number', 'N/A')}) - Category: {metadata.get('category', 'Text')}")
            st.write(doc)
            # st.json(metadata) # Uncomment to see full metadata
    else:
        st.info("No relevant text or image descriptions found in the vector database.")

    st.markdown("---")

    # --- NoSQL DB Search (Tables) ---
    st.markdown("#### Matching Tables")
    # For table search, we'll extract keywords from the query
    # In a real app, you might use an LLM to generate SQL-like queries
    query_keywords = query.lower().split() 
    table_results = search_table_db(query_keywords, num_results=3) # Adjust keywords as needed

    if table_results:
        for i, table_data in enumerate(table_results):
            st.expander(f"**Table Match {i+1}** - From {table_data.get('file_path', 'N/A')} (Page {table_data.get('page_number', 'N/A')})")
            st.components.v1.html(table_data.get('table_html', 'No HTML content'), height=300, scrolling=True)
            # st.json(table_data) # Uncomment to see full table data
    else:
        st.info("No matching tables found in the NoSQL database.")

else:
    st.info("Enter a query above to start searching your enterprise documents.")
