import os
import json
import time
import pandas as pd
from openai import OpenAI
from nltk.stem import PorterStemmer
import nltk

# =========================
# CONFIGURATION
# =========================
INPUT_FILE = "technology_misinformation_cleaned_sep.csv"
OUTPUT_FILE = "technology_misinformation_with_ai_labels.csv"

MODEL = "gpt-4o"
BATCH_SIZE = 1        # safe for rate limits
SLEEP_TIME = 0.2      # seconds between API calls

# =========================
# PROMPT
# =========================
SYSTEM_MESSAGE = (
    "You are a linguistic annotation expert. "
    "This is a stateless, single-turn task. "
    "You have NO memory of any previous inputs, outputs, or decisions. "
    "Do NOT reference or be influenced by any prior terms. "
    "Evaluate ONLY the current term provided. "
    "Follow instructions exactly and output only 0 or 1."
)

def build_prompt(term):
    return f"""
Task:
Is the following TERM ambiguous? Output ONLY a single digit: 1 or 0.

Critical rule:
Treat the TERM as a whole phrase.
Do NOT mark ambiguity just because a single word inside the TERM is ambiguous.
If the phrase itself clearly resolves the meaning, output 0.

Examples:
- "zoom" -> 1 (could be the Zoom company/app or the action)
- "zoom out" -> 0 (clearly the action)
- "apple" -> 1 (fruit vs Apple Inc.)
- "apple pie" -> 0 (clearly food)
- "cloud" -> 1 (weather cloud vs cloud computing)
- "cloud storage" -> 0 (clearly computing)

Definition:
A TERM is ambiguous if, WITHOUT any additional context beyond the TERM itself,
it can reasonably refer to two or more distinct meanings in common usage.

Output: ONLY 1 or 0. Nothing else.

TERM: "{term}"
""".strip()

# =========================
# OPENAI CLIENT
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not client.api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Restart PowerShell and VS Code.")

# =========================
# NLTK PORTER STEMMER
# =========================
stemmer = PorterStemmer()

# =========================
# MAIN LOGIC
# =========================
def main():
    print(f"Using model: {MODEL}")

    # Read CSV
    df = pd.read_csv(INPUT_FILE)

    if df.shape[1] < 2:
        raise ValueError("CSV has fewer than 2 columns – cannot read Column B.")

    # Column B (0-based index = 1)
    terms = df.iloc[:, 1].fillna("").astype(str).str.strip().tolist()
    labels = [None] * len(terms)
    stemmed_terms = [None] * len(terms)

    for i, term in enumerate(terms):
        # Handle empty cells
        if term == "":
            labels[i] = 0
            stemmed_terms[i] = ""
        else:
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    temperature=0,
                    max_tokens=2,  # limit output to 2 tokens
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": build_prompt(term)}
                    ]
                )

                result_text = response.choices[0].message.content.strip()
                
                # Extract just the digit (0 or 1)
                if '0' in result_text:
                    labels[i] = 0
                elif '1' in result_text:
                    labels[i] = 1
                else:
                    print(f"⚠️ Unexpected output for '{term}': {result_text}")
                    labels[i] = 0  # default to not ambiguous
                    
            except Exception as e:
                print(f"⚠️ Error processing '{term}': {e}")
                labels[i] = 0  # default to not ambiguous

            # Apply Porter Stemmer AFTER AI labeling
            words = term.split()
            stemmed_words = [stemmer.stem(word) for word in words]
            stemmed_terms[i] = " ".join(stemmed_words)

        if (i + 1) % 10 == 0 or (i + 1) == len(terms):
            print(f"Processed {i + 1} / {len(terms)} terms")
        
        time.sleep(SLEEP_TIME)

    # Write output
    df["AI_ambiguous"] = labels
    df["tech_stem"] = stemmed_terms
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Done.")
    print(f"Saved file: {OUTPUT_FILE}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
