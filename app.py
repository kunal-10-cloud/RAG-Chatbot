import streamlit as st
from app.ingestion import build_vector_store_from_pdf
from app.rag import answer_question

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")

st.title("PDF Question Answering Chatbot")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat" not in st.session_state:
    st.session_state.chat = []

st.sidebar.header("Upload PDF")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
)

if uploaded_pdf and st.session_state.vector_store is None:
    with st.spinner("Processing PDF..."):
        st.session_state.vector_store = build_vector_store_from_pdf(uploaded_pdf)
        st.success("PDF processed successfully")

for role, message in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(message)

question = st.chat_input("Ask a question about the PDF")

if question and st.session_state.vector_store:
    st.session_state.chat.append(("user", question))

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(st.session_state.vector_store, question)
            st.markdown(answer)

    st.session_state.chat.append(("assistant", answer))

elif question and st.session_state.vector_store is None:
    st.warning("Please upload a PDF first.")