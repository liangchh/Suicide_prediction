# Risk Scoring

This repository provides a pipeline to predict suicide thought and behavior using a pre-trained XGBoost models. It normalizes the input data, applies the models, generates risk scores, and assigns percentile-based risk groups.

## Contents

- `run_score.py`: Main script to generate scores and risk groups from the input dataset.
- `model.pkl`: Pickled dictionary containing:
  - `models`: List of trained XGBoost models.
  - `normalization`: Fitted sklearn `StandardScaler` or similar transformer.
- `dataset.csv`: Input data to be scored (must include an `ID` column).
- `requirements.txt`: Python dependencies.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Input Format
The input file must be named dataset.csv.
- It must contain an ID column.
- Other feature columns should match those expected by the model.

## How to Run
To generate risk scores and percentile groupings, run the script:

```bash
python run_score.py
```

This will:
- Load the dataset from dataset.csv.
- Drop specified columns as per dropvars from model.pkl.
- Apply normalization using the stored scaler.
- Predict risk scores using the ensemble of XGBoost models.
- Assign a RISK_GRP label based on score percentiles.


## Output
The script produces a DataFrame (df_score) with the following columns:
- `ID`: Copied from the input file.
- `Y_score`: Averaged model-predicted risk score (float).
- `RISK_GRP`: Risk group assigned based on score percentiles.

To save the results to a CSV file, you can add the following line at the end of run_score.py:

```python
df_score.to_csv('scored_output.csv', index=False)
```

## Risk Group Legend

| Risk Group              | Description         |
| ----------------------- | ------------------- |
| Top 1st percentile      | Highest risk scores |
| 2nd - 5th percentile    | Next highest        |
| ...                     | ...                 |
| 91st - 100th percentile | Lowest risk scores  |


