import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

class HeartDiseaseBayesianNetwork:
    def __init__(self):
        # Define the structure of the Bayesian Network
        self.model = BayesianNetwork([
            ('Age', 'HeartDisease'),
            ('Sex', 'HeartDisease'),
            ('ChestPain', 'HeartDisease'),
            ('BloodPressure', 'HeartDisease'),
            ('Cholesterol', 'HeartDisease'),
            ('BloodSugar', 'HeartDisease'),
            ('ECG', 'HeartDisease'),
            ('HeartRate', 'HeartDisease'),
            ('Exercise', 'HeartDisease')
        ])
        
        self.discretizers = {}

    def preprocess_data(self, data):
        """Discretize continuous variables into categorical bins"""
        continuous_vars = ['Age', 'BloodPressure', 'Cholesterol', 'HeartRate']
        
        processed_data = data.copy()
        
        for var in continuous_vars:
            discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            processed_data[var] = discretizer.fit_transform(data[var].values.reshape(-1, 1))
            self.discretizers[var] = discretizer
            
        # Convert all columns to string type for pgmpy
        processed_data = processed_data.astype(str)
        
        return processed_data

    def fit(self, data):
        """Train the Bayesian Network using the provided data"""
        # Preprocess the data
        processed_data = self.preprocess_data(data)
        
        # Estimate CPDs using Maximum Likelihood Estimation
        mle = MaximumLikelihoodEstimator(model=self.model, data=processed_data)
        
        # Estimate CPD for each node
        for node in self.model.nodes():
            cpd = mle.estimate_cpd(node)
            self.model.add_cpds(cpd)
            
        # Check if the model is valid
        assert self.model.check_model()
        
        # Initialize inference engine
        self.inference = VariableElimination(self.model)

    def predict_probability(self, patient_data):
        """Predict the probability of heart disease for a given patient"""
        # Preprocess the patient data
        processed_patient = {}
        
        for var, value in patient_data.items():
            if var in self.discretizers:
                processed_value = str(int(self.discretizers[var].transform([[value]])[0]))
            else:
                processed_value = str(value)
            processed_patient[var] = processed_value
            
        # Perform inference
        result = self.inference.query(variables=['HeartDisease'], 
                                    evidence=processed_patient)
        
        return result.values

def load_heart_disease_data():
    """Load and prepare the heart disease dataset"""
    # Load the Heart Disease dataset
    columns = ['Age', 'Sex', 'ChestPain', 'BloodPressure', 'Cholesterol', 
              'BloodSugar', 'ECG', 'HeartRate', 'Exercise', 'HeartDisease']
    
    data = pd.read_csv('heart.csv')  # You'll need to provide the correct path
    return data

# Example usage
if __name__ == "__main__":
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    synthetic_data = pd.DataFrame({
        'Age': np.random.normal(55, 10, n_samples),
        'Sex': np.random.choice(['0', '1'], n_samples),
        'ChestPain': np.random.choice(['0', '1', '2', '3'], n_samples),
        'BloodPressure': np.random.normal(130, 20, n_samples),
        'Cholesterol': np.random.normal(240, 40, n_samples),
        'BloodSugar': np.random.choice(['0', '1'], n_samples),
        'ECG': np.random.choice(['0', '1', '2'], n_samples),
        'HeartRate': np.random.normal(150, 20, n_samples),
        'Exercise': np.random.choice(['0', '1'], n_samples),
        'HeartDisease': np.random.choice(['0', '1'], n_samples)
    })
    
    # Split the data
    train_data, test_data = train_test_split(synthetic_data, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    bn_model = HeartDiseaseBayesianNetwork()
    bn_model.fit(train_data)
    
    # Example patient data
    patient = {
        'Age': 60,
        'Sex': '1',
        'ChestPain': '2',
        'BloodPressure': 140,
        'Cholesterol': 289,
        'BloodSugar': '0',
        'ECG': '1',
        'HeartRate': 150,
        'Exercise': '0'
    }
    
    # Get prediction
    probability = bn_model.predict_probability(patient)
    print("\nPatient Data:")
    for key, value in patient.items():
        print(f"{key}: {value}")
    
    print("\nPrediction Results:")
    print(f"Probability of Heart Disease: {probability[1]:.2%}")
    print(f"Probability of No Heart Disease: {probability[0]:.2%}")
    
    # Model Evaluation
    correct_predictions = 0
    total_predictions = len(test_data)
    
    for _, test_case in test_data.iterrows():
        patient_data = test_case.drop('HeartDisease').to_dict()
        prob = bn_model.predict_probability(patient_data)
        predicted_class = '1' if prob[1] > 0.5 else '0'
        if predicted_class == test_case['HeartDisease']:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    print(f"\nModel Accuracy on Test Data: {accuracy:.2%}")
