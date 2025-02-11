import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

def generate_sales_data(n_samples=1000, random_state=42):
    """Generate synthetic sales dataset with controlled randomization."""
    np.random.seed(random_state)
    
    data = {
        'product_id': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'sales_region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'customer_type': np.random.choice(['Retail', 'Wholesale', 'Online'], n_samples),
        'price': np.random.uniform(10, 500, n_samples),
        'quantity': np.random.randint(1, 100, n_samples),
        'discount_applied': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'days_since_last_purchase': np.random.randint(0, 365, n_samples)
    }
    
    # Create DataFrame first to avoid indexing issues with numpy arrays
    df = pd.DataFrame(data)
    
    # Introduce missing values using pandas
    df.loc[df.sample(frac=0.05, random_state=random_state).index, 'price'] = np.nan
    
    # Calculate total revenue after handling missing values
    df['total_revenue'] = df['price'].fillna(df['price'].median()) * df['quantity']
    
    return df

def feature_engineering(df):
    """Perform feature engineering with error handling and validation."""
    # 1. Feature Extraction with error handling
    try:
        # Handle division by zero
        df['revenue_per_unit'] = np.where(
            df['quantity'] == 0,
            0,
            df['total_revenue'] / df['quantity']
        )
        
        df['is_high_value_sale'] = (df['total_revenue'] > df['total_revenue'].quantile(0.75)).astype(int)
        
        # Ensure days_since_last_purchase is within valid range
        df['days_since_last_purchase'] = df['days_since_last_purchase'].clip(0, 365)
        
        df['purchase_frequency_segment'] = pd.cut(
            df['days_since_last_purchase'],
            bins=[0, 30, 90, 180, 365],
            labels=['Very Recent', 'Recent', 'Moderate', 'Old'],
            include_lowest=True
        )
    except Exception as e:
        raise ValueError(f"Error in feature extraction: {str(e)}")

    # 2. Feature Preprocessing
    categorical_features = ['product_id', 'sales_region', 'customer_type', 'purchase_frequency_segment']
    numerical_features = ['price', 'quantity', 'discount_applied', 'days_since_last_purchase', 'revenue_per_unit']
    
    # Validate features exist in dataframe
    missing_features = [col for col in categorical_features + numerical_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ],
        remainder='drop'
    )

    # 3. Feature Selection
    feature_selector_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=f_regression, k=10))
    ])

    # Prepare X and y with validation
    if 'total_revenue' not in df.columns:
        raise ValueError("Target variable 'total_revenue' not found in dataframe")
    
    X = df.drop('total_revenue', axis=1)
    y = df['total_revenue']

    # Validate no infinite or null values in target
    if not np.isfinite(y).all():
        raise ValueError("Target variable contains infinite or null values")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Fit and return
    feature_selector_pipeline.fit(X_train, y_train)
    
    return feature_selector_pipeline, X_train, X_test, y_train, y_test

def main():
    """Main execution with error handling and detailed reporting."""
    try:
        # Generate sales dataset
        sales_df = generate_sales_data()
        
        # Perform feature engineering
        feature_selector, X_train, X_test, y_train, y_test = feature_engineering(sales_df)
        
        # Print detailed information
        print("Data Generation and Feature Engineering Summary:")
        print("-" * 50)
        print(f"Original Dataset Shape: {sales_df.shape}")
        print(f"Number of missing values: {sales_df.isnull().sum().sum()}")
        
        print("\nFeature Statistics:")
        print("-" * 50)
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        print("\nTarget Variable Summary:")
        print("-" * 50)
        print(f"Mean revenue: ${y_train.mean():.2f}")
        print(f"Median revenue: ${y_train.median():.2f}")
        print(f"Revenue range: ${y_train.min():.2f} - ${y_train.max():.2f}")
        
        return sales_df, feature_selector
        
    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()