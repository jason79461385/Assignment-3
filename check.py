# check_models.py
import os
import google.generativeai as genai

# 確保你已經 export GOOGLE_API_KEY=...
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("❌ Can't find API Key")
else:
    genai.configure(api_key=api_key)
    print("🔍 Listing available models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")