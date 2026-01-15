from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.models import load_embeddings


def read_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


def build_vector_store_from_pdf(file):
    text = read_pdf(file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_text(text)

    embeddings = load_embeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store