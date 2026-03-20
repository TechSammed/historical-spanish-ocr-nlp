import os
from groq import Groq
from dotenv import load_dotenv

# Load env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

input_file = "results/page_1_predicted.txt"
output_file = "results/page_1_llm.txt"

# Load text
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# 🔥 Split into chunks (important)
def split_text(text, max_chars=2000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

chunks = split_text(text)

cleaned_chunks = []

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}")

    prompt = f"""
You are correcting OCR output of historical Spanish text.

IMPORTANT RULES:
- Only fix obvious OCR character errors
- Do NOT rewrite sentences
- Do NOT change word order
- Do NOT add or remove words
- Keep text as close to original as possible
- Only correct characters like:
    v ↔ u
    f ↔ s
    minor OCR mistakes

Return ONLY corrected text.

Text:
{chunk}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You fix OCR errors."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    cleaned_chunks.append(response.choices[0].message.content.strip())

# Merge chunks
final_text = " ".join(cleaned_chunks)

# Save output
with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_text)

print(f"LLM corrected text saved to: {output_file}")