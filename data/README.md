# Data

This project uses the PaySim synthetic financial transaction dataset.

## Directory Structure

```text
data/
├── raw/
├── processed/
└── sample/
    └── paysim_sample.csv
```

## Raw Data

The raw PaySim dataset is intentionally excluded from Git because of
its file size.

Download the dataset from:

https://www.kaggle.com/datasets/ealaxi/paysim1

Place the downloaded CSV at:

```
data/raw/paysim.csv
```

## Data Policy
- ```data/raw/``` contains immutable source data.
- ```data/processed/``` contains generated pipeline outputs.
- ```data/sample/``` contains small datasets used for tests and development.
- Raw and processed datasets must not be committed to Git.

The repository includes ```data/sample/paysim_sample.csv``` as a small,
version-controlled PaySim-compatible fixture for local development and
integration tests. It is not intended to represent the full dataset
distribution.

## Important PaySim Limitation

PaySim is synthetic data.

The dataset documentation notes that fraud transactions are cancelled
in the simulation. Therefore, balance fields such as:
- ```oldbalanceOrg```
- ```newbalanceOrig```
- ```oldbalanceDest```
- ```newbalanceDest```

must not be used as predictive features for the fraud model.

---
