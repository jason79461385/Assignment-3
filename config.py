import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(override=True)
if not os.getenv("GOOGLE_API_KEY"):
    print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

DATA_FOLDER = "data"
DB_FOLDER = "chroma_db"

# 檔案對應
FILES = {
    "apple": "FY24_Q4_Consolidated_Financial_Statements.pdf",
    "tesla": "tsla-20241231-gen.pdf"
}

# ==============================================================================
# Embedding Model (Can Change)
# ==============================================================================
LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embeddings():
    print(f"🔄 Loading Local Embedding Model: {LOCAL_EMBEDDING_MODEL}...")
    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)

# ==============================================================================
# 3. LLM Model (All Students should use Gemini 2.0-flash)
# ==============================================================================
def get_llm(temperature=0):
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        google_api_key=api_key,
        convert_system_message_to_human=True,
        max_output_tokens=2048
    )
    return llm