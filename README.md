# Wine origin analysis

This project uses measurements from 178 wines to answer two questions:

1. Can the measurements estimate a wine's alcohol level?
2. Can they identify its cultivar group?

The analysis prepares the data, fits a regression model, fits a classifier,
and compares a few classification models.

```bash
python -m pip install -r requirements.txt
python wine_analysis.py
```

The dataset is included with scikit-learn. The groups describe cultivars,
not wine quality.
