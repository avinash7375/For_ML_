import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

class BaggingClassifier:
    def __init__(self, base_estimator=None, n_estimators=10):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier()
        self.estimators = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.estimators = []
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            X_sample, y_sample = self.bootstrap_sample(X, y)
            # Train estimator
            estimator = clone_estimator(self.base_estimator)
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)

    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        # Majority voting
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

def clone_estimator(estimator):
    """Helper function to clone an estimator"""
    return DecisionTreeClassifier(
        criterion=estimator.criterion,
        max_depth=estimator.max_depth,
        random_state=None
    )

# Generate synthetic dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize models
base_tree = DecisionTreeClassifier(max_depth=3)
bagging_custom = BaggingClassifier(base_estimator=base_tree, n_estimators=10)
random_forest = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=10,
    random_state=42
)

# Dictionary to store results
models = {
    'Single Decision Tree': base_tree,
    'Custom Bagging': bagging_custom,
    'Random Forest': random_forest,
    'AdaBoost': adaboost
}

# Train and evaluate models
results = {}
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Plotting results
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Model Comparison - Accuracy Scores')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Example prediction
sample_data = X_test[:1]
print("\nPrediction example for first test sample:")
for name, model in models.items():
    prediction = model.predict(sample_data)
    print(f"{name}: {prediction[0]}")
