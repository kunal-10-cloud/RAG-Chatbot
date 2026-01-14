import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# =============================
# CONFIGURATION
# =============================

PDF_DIR = "data/pdfs"
VECTOR_STORE_DIR = "vector_store"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    documents = []

    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_DIR, filename)
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)
    return chunks


def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_STORE_DIR)


def main():
    print("Loading PDF documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} pages")

    print("Splitting text into chunks...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating vector embeddings and FAISS index...")
    build_vector_store(chunks)


if __name__ == "__main__":
    main()