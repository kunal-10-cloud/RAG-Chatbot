# RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) based question answering system that allows users to upload a PDF document and ask natural language questions. The system retrieves relevant content from the document and generates grounded responses strictly based on the retrieved context, with safeguards to reduce unsupported or speculative answers.

The application is built as a lightweight web interface using Streamlit and runs entirely on local resources.

## Features

- Upload and query PDF documents using natural language
- Context-aware answers powered by a Retrieval-Augmented Generation pipeline
- In-memory vector search using FAISS for fast semantic retrieval
- Hallucination mitigation through relevance and answerability checks
- Clean and minimal Streamlit-based user interface

## Tech Stack

- Python
- Streamlit
- Hugging Face Transformers (FLAN-T5)
- Sentence-Transformers
- FAISS (in-memory)
- LangChain
- PyPDF

## Running the Project Locally

### Prerequisites

- Python 3.9 or higher
- pip and virtualenv

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/RAG-Chatbot.git
   cd RAG-Chatbot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open the application**
   Open your browser at `http://localhost:8501`

## Usage

1. Upload a PDF document using the interface.
2. Wait for the document to be processed.
3. Ask questions related to the uploaded document.
4. The system will return answers based only on the document content.

If the document does not contain the required information, the system will explicitly indicate that the answer is unavailable.

## Notes

- The vector index is created in memory and is reset when the application restarts.
- The system is designed to prioritize correctness and grounded responses over speculative answers.