# Cellular network quality

This project looks at a simple product question: can network measurements
predict a poor mobile connection before a customer reports it?

The script creates a reproducible sample of 4G and 5G cell measurements,
checks the data, estimates throughput, and flags poor quality connections.
The sample is synthetic, so the project shows the method rather than making
claims about a real carrier network.

```bash
python -m pip install -r requirements.txt
python network_quality_analysis.py
```

The report includes the errors, accuracy, and the features used by the model.
