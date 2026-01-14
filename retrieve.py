from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3


def load_vector_store():

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    db = FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


def retrieve_relevant_chunks(db, query):

    results = db.similarity_search(query, k=TOP_K)
    return results


def main():
    print("Loading vector store...")
    db = load_vector_store()

    while True:
        query = input("\nEnter your question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = retrieve_relevant_chunks(db, query)

        print("\n Top relevant chunks:\n")
        for idx, doc in enumerate(results, start=1):
            print(f"--- Result {idx} ---")
            print(doc.page_content)
            print("Metadata:", doc.metadata)
            print()


if __name__ == "__main__":
    main()