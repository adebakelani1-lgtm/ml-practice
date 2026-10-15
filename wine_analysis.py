from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

wine = load_wine(as_frame=True)
data = wine.data.copy()
groups = wine.target

print("Wine origin analysis")
print("Rows:", len(data))
print("Measurements:", data.shape[1])
print("Missing values:", int(data.isna().sum().sum()))
print("Duplicate rows:", int(data.duplicated().sum()))
print("Cultivar counts:", groups.value_counts().sort_index().to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    data, groups, test_size=0.2, random_state=42, stratify=groups
)

prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
X_train_scaled = prep.fit_transform(X_train)
X_test_scaled = prep.transform(X_test)

alcohol_train = data.loc[y_train.index, "alcohol"]
alcohol_test = data.loc[y_test.index, "alcohol"]
regression = LinearRegression().fit(X_train_scaled[:, 1:], alcohol_train)
alcohol_prediction = regression.predict(X_test_scaled[:, 1:])
print("Alcohol estimate MAE:", round(mean_absolute_error(alcohol_test, alcohol_prediction), 3))

classifier = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, random_state=42
).fit(X_train_scaled, y_train)
group_prediction = classifier.predict(X_test_scaled)
print("Cultivar classifier accuracy:", round(accuracy_score(y_test, group_prediction), 3))

models = {
    "Decision tree": DecisionTreeClassifier(
        criterion="entropy", max_depth=3, random_state=42
    ),
    "Logistic regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "SVM": make_pipeline(StandardScaler(), SVC()),
}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Cross validation:")
for name, model in models.items():
    score = cross_val_score(model, data, groups, cv=folds, scoring="accuracy")
    print(f"  {name}: {score.mean():.3f} +/- {score.std():.3f}")
