import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# EMBEDDINGS CONFIG
# =========================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# =========================
# LLM CONFIG (BASE + LoRA)
# =========================

BASE_MODEL = "google/flan-t5-base"
LORA_ADAPTER_PATH = "finetuned-financial-flan"

def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=None
    )

    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH
    )

    model.eval()
    return tokenizer, model