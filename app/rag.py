import torch
from app.models import load_llm

TOP_K = 3
REFUSAL_MESSAGE = "I do not have enough information to answer this question."


def retrieve_context(vector_store, question):
    docs = vector_store.similarity_search(question, k=TOP_K)
    return "\n\n".join(doc.page_content for doc in docs)


def is_context_relevant(context: str, question: str) -> bool:


    question_keywords = [
        word.lower()
        for word in question.split()
        if len(word) > 3
    ]

    context_lower = context.lower()

    return any(keyword in context_lower for keyword in question_keywords)


def build_prompt(context, question):
    return f"""
You are a financial policy assistant.

Answer the question strictly using the context below.
Use a complete sentence.
Do not answer with a standalone number.
If the answer is not present in the context, say:
"{REFUSAL_MESSAGE}"

Context:
{context}

Question:
{question}

Answer:
""".strip()


def answer_question(vector_store, question):
    tokenizer, model = load_llm()

    context = retrieve_context(vector_store, question)

 
    if not context.strip():
        return REFUSAL_MESSAGE

    
    if not is_context_relevant(context, question):
        return REFUSAL_MESSAGE

    prompt = build_prompt(context, question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)