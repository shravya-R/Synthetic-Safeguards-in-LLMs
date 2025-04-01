import ast
import random
from typing import Dict
import pandas as pd
from datasets import load_dataset
import json
from groq import Groq
import time
from tqdm import tqdm

client = Groq(
    api_key="-----"
)

def generate_instruction(document_type, document_description, expanded_description):
    prompt = f"""Given this financial document info:
    Type: {document_type}
    Description: {document_description}
    Expanded: {expanded_description}

    Create a brief, one-sentence instruction to generate this document.

    Instruction:"""

    return prompt

def main():
    mapping_df = pd.read_csv("synthetic_text_full.csv")
    output_path = "instruction_tuning_dataset.csv"
    data = load_dataset('LightFury9/gretelai_synthetic_pii_finance_english')
    train_data = data['cleaned'].sort("index")
    train_data = train_data.to_pandas()
    train_data = train_data[0:750]

    instruction_tuning_data = []
    for i, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="Processing rows"):
        prompt = generate_instruction(
        row['document_type'],
        row['document_description'],
        row['expanded_description']
    )
        while True:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="llama3-8b-8192"
                )
                instruction = chat_completion.choices[0].message.content
                synthetic_text = mapping_df.loc[mapping_df['generated_text'] == row['generated_text'], 'synthetic_text'].values
                if len(synthetic_text) > 0:
                    output_text = synthetic_text[0]
                else:
                    output_text = row['generated_text']    
                instruction_tuning_data.append({
                    "instruction": instruction,
                    "output": output_text
                })                                 
                break

            except Exception as e:
                print(f"Error encountered: {e}. Retrying in 1 minute...")
                time.sleep(60)
    df = pd.DataFrame(instruction_tuning_data)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
