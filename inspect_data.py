
import pandas as pd
import os

def inspect_excel(filepath):
    try:
        df = pd.read_excel(filepath)
        print(f"\n--- {os.path.basename(filepath)} ---")
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print("Head:\n", df.head())
        print("Info:\n")
        df.info()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

def read_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            print(f"\n--- {os.path.basename(filepath)} ---")
            print(f.read())
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    inspect_excel("data/p33.xlsx")
    read_text_file("docs/Resultados comparativos.txt")
