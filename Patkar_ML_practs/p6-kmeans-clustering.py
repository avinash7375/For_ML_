import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class KMeansClustering:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        # Randomly initialize centroids
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Assign points to nearest centroid
            distances = self._calculate_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = cluster_points.mean(axis=0)
            
            # Check convergence
            if np.all(old_centroids == self.centroids):
                break
                
    def predict(self, X):
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)
    
    def _calculate_distances(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances

def create_sample_data(n_samples=300):
    """Create sample dataset with known clusters"""
    X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=42)
    return X, y

def plot_clusters(X, labels, centroids, title):
    """Visualize clustering results"""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title(title)
    plt.colorbar(scatter)
    plt.legend()
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

def main():
    # Create sample dataset
    X, y_true = create_sample_data()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train KMeans model
    kmeans = KMeansClustering(k=3)
    kmeans.fit(X_train_scaled)
    
    # Make predictions
    train_predictions = kmeans.predict(X_train_scaled)
    test_predictions = kmeans.predict(X_test_scaled)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    silhouette_avg = silhouette_score(X_train_scaled, train_predictions)
    
    # Print results
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    print(f"Silhouette Score: {silhouette_avg:.2f}")
    
    # Visualize results
    plot_clusters(X_train_scaled, train_predictions, kmeans.centroids, 
                 'K-Means Clustering Results (Training Data)')
    plot_confusion_matrix(y_test, test_predictions)
    
    # Additional analysis: Within-cluster sum of squares (WCSS)
    wcss = sum(np.min(kmeans._calculate_distances(X_train_scaled), axis=1))
    print(f"\nWithin-cluster Sum of Squares: {wcss:.2f}")

if __name__ == "__main__":
    main()
