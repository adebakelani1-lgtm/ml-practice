# Results

The first run found 178 rows, 13 measurements, no missing values, and no
duplicate rows. The three cultivar groups had 59, 71, and 48 wines.

The linear model estimated alcohol within about 0.43 percentage points on the
held-out set. The entropy based tree identified the cultivar group with 97.2%
accuracy on that same split.

In five fold cross validation, logistic regression and SVM both reached 98.3%,
while the tree reached 90.4%. Scaling helped the models
that use distances or a linear boundary. These scores describe this small
dataset and should not be read as a test of wine quality.
