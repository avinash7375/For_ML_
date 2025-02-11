import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, random_state=42):
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and train the SVM model
        self.model = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        # Scale the features using the same scaler
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_support_vectors(self):
        return self.model.support_vectors_

def load_and_prepare_data():
    """Load breast cancer dataset and prepare it for training"""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    return X, y, feature_names, target_names

def perform_grid_search(X_train, y_train):
    """Perform grid search to find optimal hyperparameters"""
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    grid_search = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

def plot_confusion_matrix(y_true, y_pred, target_names):
    """Plot confusion matrix with annotations"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_feature_importance(coefficients, feature_names):
    """Plot feature importance for linear kernel"""
    importance = np.abs(coefficients)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance)), importance)
    plt.xticks(range(len(importance)), feature_names, rotation=45, ha='right')
    plt.title('Feature Importance (Absolute Values of Coefficients)')
    plt.tight_layout()
    plt.show()

def main():
    # Load and prepare data
    X, y, feature_names, target_names = load_and_prepare_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Perform grid search for optimal parameters
    print("Performing grid search for optimal parameters...")
    best_params, best_score = perform_grid_search(X_train, y_train)
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    # Train SVM with best parameters
    svm = SVMClassifier(
        kernel=best_params['kernel'],
        C=best_params['C']
    )
    svm.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)
    
    # Print classification reports
    print("\nTraining Set Performance:")
    print(classification_report(y_train, y_train_pred, target_names=target_names))
    
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_test_pred, target_names)
    
    # If using linear kernel, plot feature importance
    if best_params['kernel'] == 'linear':
        plot_feature_importance(svm.model.coef_[0], feature_names)
    
    # Additional analysis
    support_vectors_count = len(svm.get_support_vectors())
    print(f"\nNumber of support vectors: {support_vectors_count}")
    
    # Plot decision regions for two selected features
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap='viridis')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Decision Regions (First Two Features)')
    plt.colorbar(label='Predicted Class')
    plt.show()

if __name__ == "__main__":
    main()
