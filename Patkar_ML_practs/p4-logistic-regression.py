import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize Logistic Regression Classifier
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            num_iterations (int): Number of training iterations
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()
        
    def generate_sample_data(self, n_samples=1000):
        """
        Generate synthetic dataset and save to CSV
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            str: Path to generated CSV file
        """
        np.random.seed(42)
        
        # Generate two features
        X1 = np.random.randn(n_samples)
        X2 = np.random.randn(n_samples)
        
        # Generate target variable (binary classification)
        # Decision boundary: 2*X1 + X2 - 1 = 0
        y = (2 * X1 + X2 - 1 > 0).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature1': X1,
            'feature2': X2,
            'target': y
        })
        
        # Save to CSV
        csv_path = 'logistic_regression_data.csv'
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def load_data(self, csv_path):
        """
        Load and preprocess data from CSV file
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            tuple: Features (X) and target (y)
        """
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Separate features and target
        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        return X, y
    
    def sigmoid(self, z):
        """
        Compute sigmoid function
        
        Args:
            z (array): Input values
            
        Returns:
            array: Sigmoid of input values
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Train logistic regression model
        
        Args:
            X (array): Training features
            y (array): Target values
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize parameters
        n_samples, n_features = X_scaled.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training history for visualization
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Forward pass
            linear_model = np.dot(X_scaled, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Compute cost
            cost = (-1/n_samples) * np.sum(
                y * np.log(y_predicted + 1e-15) + 
                (1 - y) * np.log(1 - y_predicted + 1e-15)
            )
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X_scaled.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """
        Make predictions for new data
        
        Args:
            X (array): Input features
            
        Returns:
            array: Predicted classes
        """
        X_scaled = self.scaler.transform(X)
        linear_model = np.dot(X_scaled, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Args:
            X (array): Test features
            y (array): True target values
        """
        # Make predictions
        y_pred = self.predict(X)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.title('Training Cost History')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        
        # Plot decision boundary (if 2D)
        if X.shape[1] == 2:
            plt.figure(figsize=(10, 6))
            
            # Create mesh grid
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                np.arange(y_min, y_max, 0.1))
            
            # Make predictions for mesh grid points
            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary and data points
            plt.contourf(xx, yy, Z, alpha=0.4)
            plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
            plt.title('Decision Boundary')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()

def main():
    # Create instance of LogisticRegression
    model = LogisticRegressionClassifier(learning_rate=0.01, num_iterations=1000)
    
    # Generate and load sample data
    print("Generating sample data...")
    csv_path = model.generate_sample_data()
    X, y = model.load_data(csv_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("\nTraining logistic regression model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    model.evaluate(X_test, y_test)
    
    # Example prediction
    print("\nExample predictions:")
    example_data = np.array([[1.5, 0.5], [-0.5, -1.0]])
    predictions = model.predict(example_data)
    for features, pred in zip(example_data, predictions):
        print(f"Features: {features}, Predicted class: {pred}")

if __name__ == "__main__":
    main()
