import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class LinearRegressionAnalysis:
    def __init__(self):
        """
        Initialize the Linear Regression Analysis class
        with methods for both simple and multiple linear regression
        """
        self.simple_model = None
        self.multiple_model = None
        self.scaler = StandardScaler()

    def generate_synthetic_data(self, n_samples=200, simple=True):
        """
        Generate synthetic datasets for regression analysis
        
        Args:
            n_samples (int): Number of data points
            simple (bool): Whether to generate simple or multiple regression dataset
        
        Returns:
            tuple: Features (X) and target variable (y)
        """
        np.random.seed(42)
        
        if simple:
            # Simple Linear Regression Dataset
            X = np.linspace(0, 10, n_samples).reshape(-1, 1)
            y = 2 * X.ravel() + 1 + np.random.normal(0, 1, n_samples)
            return X, y.ravel()
        else:
            # Multiple Linear Regression Dataset
            X = np.random.rand(n_samples, 3)  # 3 features
            true_coef = np.array([1.5, -0.8, 2.3])
            noise = np.random.normal(0, 0.5, n_samples)
            y = np.dot(X, true_coef) + 1.0 + noise
            return X, y

    def simple_linear_regression(self):
        """
        Perform Simple Linear Regression
        """
        # Generate data
        X, y = self.generate_synthetic_data(simple=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train the model
        self.simple_model = LinearRegression()
        self.simple_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.simple_model.predict(X_test)
        
        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual Data')
        plt.plot(X_test, y_pred, color='red', label='Regression Line')
        plt.title('Simple Linear Regression')
        plt.xlabel('Input Feature')
        plt.ylabel('Target Variable')
        plt.legend()
        plt.show()
        
        # Print results
        print("\nSimple Linear Regression Results:")
        print(f"Intercept: {self.simple_model.intercept_:.4f}")
        print(f"Coefficient: {self.simple_model.coef_[0]:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared Score: {r2:.4f}")
        
        return self.simple_model

    def multiple_linear_regression(self):
        """
        Perform Multiple Linear Regression
        """
        # Generate data
        X, y = self.generate_synthetic_data(simple=False)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train the model
        self.multiple_model = LinearRegression()
        self.multiple_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.multiple_model.predict(X_test_scaled)
        
        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Visualization of predicted vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('Multiple Linear Regression: Predicted vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()
        
        # Print results
        print("\nMultiple Linear Regression Results:")
        print("Coefficients:")
        for i, coef in enumerate(self.multiple_model.coef_):
            print(f"Feature {i+1}: {coef:.4f}")
        print(f"Intercept: {self.multiple_model.intercept_:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared Score: {r2:.4f}")
        
        return self.multiple_model

    def predict(self, X, model_type='simple'):
        """
        Make predictions using the trained model
        
        Args:
            X (array-like): Input features
            model_type (str): 'simple' or 'multiple'
        
        Returns:
            array: Predictions
        """
        if model_type == 'simple':
            if self.simple_model is None:
                raise ValueError("Simple Linear Regression model not trained. Run simple_linear_regression() first.")
            return self.simple_model.predict(X)
        elif model_type == 'multiple':
            if self.multiple_model is None:
                raise ValueError("Multiple Linear Regression model not trained. Run multiple_linear_regression() first.")
            X_scaled = self.scaler.transform(X)
            return self.multiple_model.predict(X_scaled)
        else:
            raise ValueError("Invalid model type. Choose 'simple' or 'multiple'.")

def main():
    # Create an instance of the Linear Regression Analysis
    lr_analysis = LinearRegressionAnalysis()
    
    print("Performing Simple Linear Regression:")
    simple_model = lr_analysis.simple_linear_regression()
    
    print("\n" + "="*50 + "\n")
    
    print("Performing Multiple Linear Regression:")
    multiple_model = lr_analysis.multiple_linear_regression()

if __name__ == "__main__":
    main()