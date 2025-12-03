import pandas as pd
from pypdf import PdfReader
import os

def read_pdf(file_path):
    print(f"\n--- Reading {os.path.basename(file_path)} ---")
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(text[:2000]) # Print first 2000 chars to avoid overwhelming output
        if len(text) > 2000:
            print("\n... [Truncated] ...")
    except Exception as e:
        print(f"Error reading PDF: {e}")

def read_excel(file_path):
    print(f"\n--- Reading {os.path.basename(file_path)} ---")
    try:
        df = pd.read_excel(file_path)
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nInfo:")
        print(df.info())
    except Exception as e:
        print(f"Error reading Excel: {e}")

base_path = "/home/gustavin/Downloads/Izaac"
files = [
    "PROVA_2UP_MACHINE-LEARNING_2025_2.pdf",
    "M.E 2ªUP Machine Learning (1).pdf",
    "p33.xlsx"
]

for f in files:
    read_pdf(os.path.join(base_path, f)) if f.endswith('.pdf') else read_excel(os.path.join(base_path, f))
