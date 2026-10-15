# Results

The sample has 600 connection records. Eighteen latency values were missing,
so the model filled them with the training median. The poor quality label is a
simple product rule based on low throughput or high packet loss.

The first run estimated throughput with a mean absolute error of about 7 Mbps.
The classifier reached 91.7% accuracy and an ROC AUC of 0.977. The
largest throughput effects came from bandwidth, signal strength, cell load,
and latency.

This is a synthetic project. A real network team would replace the generator
with timestamped cell data, check the label with customer experience measures,
and evaluate performance across locations and time periods.
