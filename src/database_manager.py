import os
from pymongo import MongoClient
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv() # Load environment variables

# --- MongoDB Setup (NoSQL for Tables) ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "enterprise_knowledge")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
table_collection = mongo_db["tables"]

# --- ChromaDB Setup (Vector DB for Text and Image Descriptions) ---
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pdf_chunks")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)
chroma_client = PersistentClient(path=CHROMA_DB_PATH)

try:
    vector_collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=openai_ef)
except:
    vector_collection = chroma_client.create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=openai_ef)


def store_elements_in_databases(elements):
    """
    Stores processed elements into the appropriate database.
    """
    text_contents = []
    text_metadatas = []
    text_ids = []

    for el in elements:
        if el["type"] == "Table":
            # Store tables in MongoDB
            table_collection.insert_one({
                "element_id": el["metadata"].element_id,
                "file_path": getattr(el["metadata"], 'filename', 'unknown'),
                "page_number": getattr(el["metadata"], 'page_number', None),
                "table_html": el["content"],
                "text_content": getattr(el["metadata"], 'text_content', el["content"]), # Fallback
                "category": el["type"],
                "source": getattr(el["metadata"], 'url', 'local_file')
            })
            print(f"  Stored Table {el['metadata'].element_id} in MongoDB.")
        elif el["type"] == "Text" or el["type"] == "Image":
            # Prepare text and image descriptions for ChromaDB
            text_contents.append(el["content"])
            # Ensure metadata is serializable and relevant
            metadata_dict = {
                "file_path": getattr(el["metadata"], 'filename', 'unknown'),
                "page_number": getattr(el["metadata"], 'page_number', None),
                "category": el["type"],
                "source": getattr(el["metadata"], 'url', 'local_file')
            }
            if el["type"] == "Image":
                 metadata_dict["image_path"] = getattr(el["metadata"], 'image_path', None)

            text_metadatas.append(metadata_dict)
            text_ids.append(el["metadata"].element_id)
            print(f"  Prepared {el['type']} {el['metadata'].element_id} for ChromaDB.")
    
    if text_contents:
        # Add to ChromaDB in batches
        # ChromaDB requires lists for documents, metadatas, and ids
        vector_collection.add(
            documents=text_contents,
            metadatas=text_metadatas,
            ids=text_ids
        )
        print(f"  Stored {len(text_contents)} Text/Image elements in ChromaDB.")

def search_vector_db(query_text, num_results=5):
    """
    Searches the Vector DB for relevant text chunks.
    """
    results = vector_collection.query(
        query_texts=[query_text],
        n_results=num_results,
        include=['documents', 'metadatas', 'distances']
    )
    return results

def search_table_db(query_keywords, num_results=5):
    """
    Searches the NoSQL DB for tables containing keywords.
    This is a basic keyword search. For advanced querying,
    you'd integrate a more sophisticated NoSQL query builder or LLM.
    """
    regex_pattern = ".*" + ".*|".join(query_keywords) + ".*"
    results = list(table_collection.find(
        {"table_html": {"$regex": regex_pattern, "$options": "i"}}
    ).limit(num_results))
    return results
