from sklearn.datasets import load_wine
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

wine = load_wine(as_frame=True)
# Alcohol is the target, so remove it from the inputs.
X = wine.data.drop(columns="alcohol")
y = wine.data["alcohol"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Linear regression": LinearRegression(),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\n" + name)
    print("Train MAE:", round(mean_absolute_error(y_train, model.predict(X_train)), 3))
    print("Test MAE:", round(mean_absolute_error(y_test, predictions), 3))
    print("Test MSE:", round(mean_squared_error(y_test, predictions), 3))
    print("Test R2:", round(r2_score(y_test, predictions), 3))

print("\nFirst five predictions:")
for actual, predicted in zip(y_test.iloc[:5], predictions[:5]):
    print(f"Actual: {actual:.2f}, predicted: {predicted:.2f}")
