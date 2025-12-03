# Machine Learning Assignment - Churn Analysis

This repository contains the Python implementation for the Machine Learning assignment on Churn Prediction.

## Project Structure

- `churn_analysis.py`: Main script to load data, preprocess, train models, and evaluate results.
- `inspect_data.py`: Utility script to inspect the input data.
- `data/`: Contains the dataset (`p33.xlsx`).
- `docs/`: Contains assignment instructions and reference materials.
- `results/`: Contains the output metrics (`python_results.csv`).
- `requirements.txt`: Python dependencies.

## Setup

1.  Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the analysis script:
```bash
python3 churn_analysis.py
```

The script will:
1.  Load the data from `data/p33.xlsx`.
2.  Preprocess the data (StandardScaler, OneHotEncoder).
3.  Train and evaluate 7 models using 10-fold Cross-Validation.
4.  Print the results table to the console.
5.  Save the results to `results/python_results.csv`.

## Results Summary

The analysis confirms that **Logistic Regression** and **Gradient Boosting** are the top-performing models, with **SVM** showing significant improvement after proper scaling in Python.
