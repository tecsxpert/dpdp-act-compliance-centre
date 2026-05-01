import os
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize ChromaDB persistent client
# This saves the vector data locally so it survives server restarts
CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'chroma_data')
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Create or connect to the collection
collection = chroma_client.get_or_create_collection(name="dpdp_compliance_docs")

# 2. Load the embedding model
# all-MiniLM-L6-v2 is the standard, fast, and highly efficient sentence-transformer model
print("Loading embedding model... (this may take a few seconds on first run)")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Slices text into chunks of `chunk_size` characters with an `overlap`.
    Required by Day 5 Capstone guide.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move the start pointer forward, minus the overlap
        start += chunk_size - overlap
        
    return chunks

def ingest_document(doc_id, text, metadata=None):
    """
    Chunks a document, generates vector embeddings, and stores them in ChromaDB.
    """
    if metadata is None:
        metadata = {"source": "manual_ingestion"}

    # 1. Chunk the text
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    
    # 2. Generate embeddings for all chunks at once
    embeddings = embedding_model.encode(chunks).tolist()
    
    # 3. Prepare IDs and Metadata for each chunk
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [metadata for _ in range(len(chunks))]
    
    # 4. Store in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully ingested document '{doc_id}' into {len(chunks)} chunks.")

# --- Quick Test Block ---
if __name__ == '__main__':
    # A dummy DPDP Act rule to test our pipeline
    test_document = (
        "The Digital Personal Data Protection Act (DPDP Act) of India mandates that Data Fiduciaries "
        "must obtain verifiable parental consent before processing any personal data of a child (a person under 18). "
        "Furthermore, fiduciaries are prohibited from undertaking tracking or behavioral monitoring of children "
        "or targeted advertising directed at children. In the event of a personal data breach, the Data Fiduciary "
        "must intimate the Data Protection Board of India and each affected Data Principal. Penalties for non-compliance "
        "can reach up to 250 crore rupees per instance."
    )
    
    print("Testing RAG Ingestion Pipeline...")
    ingest_document(doc_id="dpdp_rule_001", text=test_document, metadata={"category": "children_and_breaches"})
    
    # Verify it saved
    count = collection.count()
    print(f"Total chunks now stored in ChromaDB: {count}")