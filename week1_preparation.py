from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wine = load_wine(as_frame=True)
X = wine.data.copy()
y = wine.target

print("Rows and features:", X.shape)
print("Missing values:", X.isna().sum().sum())
print("Duplicate rows:", X.duplicated().sum())
print("Group counts:\n", y.value_counts().sort_index())
print("Feature summary:\n", X.describe().round(2))

# Add missing values just for practice.
X.loc[[0, 10, 20, 30], "magnesium"] = float("nan")
print("Practice missing values:", X.isna().sum().sum())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Learn the median and scale from training data.
imputer = SimpleImputer(strategy="median")
train_filled = imputer.fit_transform(X_train)
test_filled = imputer.transform(X_test)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_filled)
test_scaled = scaler.transform(test_filled)

print("Training shape:", train_scaled.shape)
print("Test shape:", test_scaled.shape)
print("Training means after scaling:", train_scaled.mean(axis=0).round(2))
