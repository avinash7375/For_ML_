import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class TreeClassifier:
    def __init__(self, classifier_type='rf', n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize the classifier
        classifier_type: 'dt' for Decision Tree, 'rf' for Random Forest
        """
        self.classifier_type = classifier_type
        self.random_state = random_state
        
        if classifier_type == 'dt':
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Train the model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance scores"""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

def load_and_prepare_data():
    """Load breast cancer dataset and prepare it for training"""
    data = load_breast_cancer()
    return (
        data.data,
        data.target,
        data.feature_names,
        data.target_names
    )

def plot_confusion_matrix(y_true, y_pred, target_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_feature_importance(importance_df, title):
    """Plot feature importance"""
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importance_df.head(10))
    plt.title(f'Top 10 Feature Importance - {title}')
    plt.show()

def visualize_decision_tree(model, feature_names, target_names):
    """Visualize the decision tree"""
    plt.figure(figsize=(20,10))
    plot_tree(model, feature_names=feature_names, 
              class_names=target_names, filled=True, rounded=True)
    plt.title('Decision Tree Visualization')
    plt.show()

def plot_learning_curves(model, X, y, title):
    """Plot learning curves using cross-validation"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []
    
    for size in train_sizes:
        X_subset, _, y_subset, _ = train_test_split(
            X, y, train_size=size, random_state=42
        )
        scores = cross_val_score(model, X_subset, y_subset, cv=5)
        train_scores.append(scores.mean())
        val_scores.append(scores.std())
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, label='Training score')
    plt.fill_between(train_sizes, 
                     np.array(train_scores) - np.array(val_scores),
                     np.array(train_scores) + np.array(val_scores), 
                     alpha=0.1)
    plt.title(f'Learning Curves - {title}')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def main():
    # Load data
    X, y, feature_names, target_names = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train Decision Tree
    dt_classifier = TreeClassifier(classifier_type='dt', max_depth=5)
    dt_classifier.fit(X_train, y_train)
    
    # Initialize and train Random Forest
    rf_classifier = TreeClassifier(classifier_type='rf', n_estimators=100, max_depth=5)
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    dt_pred = dt_classifier.predict(X_test)
    rf_pred = rf_classifier.predict(X_test)
    
    # Print results for Decision Tree
    print("\nDecision Tree Results:")
    print("----------------------")
    print(f"Accuracy: {accuracy_score(y_test, dt_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, dt_pred, target_names=target_names))
    
    # Print results for Random Forest
    print("\nRandom Forest Results:")
    print("----------------------")
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred, target_names=target_names))
    
    # Plot confusion matrices
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(y_test, dt_pred, target_names)
    plt.title('Decision Tree - Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(y_test, rf_pred, target_names)
    plt.title('Random Forest - Confusion Matrix')
    
    # Plot feature importance
    dt_importance = dt_classifier.get_feature_importance(feature_names)
    rf_importance = rf_classifier.get_feature_importance(feature_names)
    
    plot_feature_importance(dt_importance, 'Decision Tree')
    plot_feature_importance(rf_importance, 'Random Forest')
    
    # Visualize decision tree (only for Decision Tree classifier)
    visualize_decision_tree(dt_classifier.model, feature_names, target_names)
    
    # Plot learning curves
    plot_learning_curves(dt_classifier.model, X, y, 'Decision Tree')
    plot_learning_curves(rf_classifier.model, X, y, 'Random Forest')

if __name__ == "__main__":
    main()
