from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

wine = load_wine(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
)
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print("Baseline accuracy:", round(baseline.score(X_test, y_test), 3))

tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
print("Train accuracy:", round(tree.score(X_train, y_train), 3))
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print("Confusion matrix: rows = actual, columns = predicted")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, target_names=wine.target_names))
print(export_text(tree, feature_names=list(wine.data.columns)))
