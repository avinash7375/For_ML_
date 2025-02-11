import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

class HierarchicalClustering:
    def __init__(self, n_clusters=3, linkage_method='ward'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.labels_ = None
        self.linkage_matrix = None
        
    def fit(self, X):
        """Fit the hierarchical clustering model"""
        # Compute the linkage matrix
        self.linkage_matrix = linkage(X, method=self.linkage_method)
        # Get cluster labels
        self.labels_ = fcluster(self.linkage_matrix, self.n_clusters, criterion='maxclust')
        self.labels_ = self.labels_ - 1  # Convert to 0-based indexing
        return self
    
    def predict(self, X):
        """Predict clusters for new data points"""
        if self.linkage_matrix is None:
            raise Exception("Model must be fitted before making predictions")
        
        # For each new point, find the closest trained point and use its cluster
        predictions = []
        for point in X:
            distances = np.sqrt(np.sum((self.X_train - point) ** 2, axis=1))
            closest_point_idx = np.argmin(distances)
            predictions.append(self.labels_[closest_point_idx])
        
        return np.array(predictions)
    
    def fit_predict(self, X):
        """Fit the model and return cluster labels"""
        self.X_train = X
        self.fit(X)
        return self.labels_

def plot_dendrogram(linkage_matrix):
    """Plot the dendrogram"""
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

def plot_clusters(X, labels, title):
    """Plot clustering results"""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def create_sample_data(n_samples=300):
    """Create sample dataset with known clusters"""
    X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=42)
    return X, y

def evaluate_clustering(X, labels, y_true):
    """Calculate clustering evaluation metrics"""
    accuracy = accuracy_score(y_true, labels)
    silhouette = silhouette_score(X, labels)
    return accuracy, silhouette

def main():
    # Create sample dataset
    X, y_true = create_sample_data()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train hierarchical clustering model
    hc = HierarchicalClustering(n_clusters=3)
    train_predictions = hc.fit_predict(X_train_scaled)
    
    # Make predictions on test set
    test_predictions = hc.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    train_accuracy, train_silhouette = evaluate_clustering(X_train_scaled, train_predictions, y_train)
    test_accuracy, test_silhouette = evaluate_clustering(X_test_scaled, test_predictions, y_test)
    
    # Print results
    print("Training Results:")
    print(f"Accuracy: {train_accuracy:.2f}")
    print(f"Silhouette Score: {train_silhouette:.2f}")
    print("\nTesting Results:")
    print(f"Accuracy: {test_accuracy:.2f}")
    print(f"Silhouette Score: {test_silhouette:.2f}")
    
    # Visualizations
    plot_dendrogram(hc.linkage_matrix)
    plot_clusters(X_train_scaled, train_predictions, 'Hierarchical Clustering Results (Training Data)')
    plot_confusion_matrix(y_test, test_predictions)
    
    # Additional Analysis: Cophenetic Correlation Coefficient
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    
    c, coph_dists = cophenet(hc.linkage_matrix, pdist(X_train_scaled))
    print(f"\nCophenetic Correlation Coefficient: {c:.2f}")

if __name__ == "__main__":
    main()
