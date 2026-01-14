import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


VECTOR_STORE_DIR = "vector_store"

EMBEDDING_MODEL_PATH = (
    "/Users/kunalll/.cache/huggingface/hub/"
    "models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)

LLM_MODEL = "google/flan-t5-base"
TOP_K = 3



def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db



def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,
        local_files_only=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL,
        local_files_only=True,
    )

    model.eval()
    return tokenizer, model



def retrieve_context(db, question):
    docs = db.similarity_search(question, k=TOP_K)
    return "\n\n".join(doc.page_content for doc in docs)



def build_prompt(context, question):
    return f"""
You are a professional financial advisor chatbot.

Using ONLY the information provided in the context below:
- Answer in complete sentences
- Maintain a professional and helpful tone
- Briefly explain the answer when relevant
- Do NOT add any information not present in the context

If the answer is not available in the context, say:
"I do not have enough information to answer this question."

Context:
{context}

Question:
{question}

Answer (as a financial advisor):
""".strip()



def generate_answer(tokenizer, model, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,        
            temperature=0.6,       
            top_p=0.9,             
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def main():
    print(" Loading vector store... ")
    db = load_vector_store()

    print(" Loading language model... ")
    tokenizer, model = load_llm()

    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        context = retrieve_context(db, question)
        prompt = build_prompt(context, question)
        answer = generate_answer(tokenizer, model, prompt)

        print("\n Answer:\n")
        print(answer)


if __name__ == "__main__":
    main()