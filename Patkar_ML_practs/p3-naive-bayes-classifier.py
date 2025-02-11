import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

class NaiveBayesDocumentClassifier:
    def __init__(self):
        """
        Initialize the Naive Bayes Document Classifier
        """
        self.classes = None
        self.class_priors = None
        self.feature_probs = None
        self.vectorizer = CountVectorizer(stop_words='english')
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic document dataset for classification
        
        Args:
            n_samples (int): Number of documents to generate
            
        Returns:
            tuple: Documents (X) and their classifications (y)
        """
        np.random.seed(42)
        
        # Define document categories and their characteristic words
        categories = {
            'technology': ['computer', 'software', 'internet', 'data', 'programming',
                          'algorithm', 'network', 'digital', 'system', 'code'],
            'sports': ['football', 'basketball', 'soccer', 'game', 'player',
                      'team', 'score', 'win', 'tournament', 'championship'],
            'business': ['market', 'finance', 'investment', 'company', 'stock',
                        'business', 'economy', 'trade', 'profit', 'corporate']
        }
        
        documents = []
        labels = []
        
        for _ in range(n_samples):
            # Randomly select a category
            category = np.random.choice(list(categories.keys()))
            
            # Generate a synthetic document
            n_words = np.random.randint(20, 50)
            category_words = categories[category]
            other_words = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'in',
                          'on', 'at', 'to', 'for', 'of', 'with', 'by']
            
            # Mix category-specific words with common words
            doc_words = (np.random.choice(category_words, size=n_words//2).tolist() +
                        np.random.choice(other_words, size=n_words//2).tolist())
            np.random.shuffle(doc_words)
            
            document = ' '.join(doc_words)
            documents.append(document)
            labels.append(category)
            
        return documents, labels

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        
        Args:
            X (list): List of documents
            y (list): List of document classifications
        """
        # Convert documents to term frequency matrix
        X_freq = self.vectorizer.fit_transform(X)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get unique classes and their probabilities
        self.classes = np.unique(y)
        self.class_priors = {}
        self.feature_probs = {}
        
        n_docs = len(y)
        
        # Calculate class priors and feature probabilities
        for c in self.classes:
            # Prior probabilities P(c)
            self.class_priors[c] = np.sum(np.array(y) == c) / n_docs
            
            # Get documents for this class
            class_docs = X_freq[np.array(y) == c].toarray()
            
            # Calculate feature probabilities with Laplace smoothing
            feature_counts = np.sum(class_docs, axis=0) + 1  # Add 1 for Laplace smoothing
            total_words = np.sum(feature_counts)
            
            self.feature_probs[c] = {
                feature: count/total_words
                for feature, count in zip(feature_names, feature_counts)
            }
    
    def predict(self, X):
        """
        Predict classifications for new documents
        
        Args:
            X (list): List of documents to classify
            
        Returns:
            list: Predicted classifications
        """
        X_freq = self.vectorizer.transform(X)
        feature_names = self.vectorizer.get_feature_names_out()
        
        predictions = []
        
        for doc in X_freq:
            # Calculate probability for each class
            class_scores = {}
            
            for c in self.classes:
                # Start with log of class prior
                score = np.log(self.class_priors[c])
                
                # Add log probabilities of features
                for feature, count in zip(feature_names, doc.toarray()[0]):
                    if count > 0:
                        score += count * np.log(self.feature_probs[c][feature])
                
                class_scores[c] = score
            
            # Select class with highest probability
            predictions.append(max(class_scores.items(), key=lambda x: x[1])[0])
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier's performance
        
        Args:
            X_test (list): Test documents
            y_test (list): True classifications
        """
        y_pred = self.predict(X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Calculate and print accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Visualize class distribution
        plt.figure(figsize=(10, 6))
        pd.Series(y_test).value_counts().plot(kind='bar')
        plt.title('Distribution of Document Classes in Test Set')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    # Create an instance of the classifier
    classifier = NaiveBayesDocumentClassifier()
    
    # Generate synthetic document dataset
    print("Generating synthetic document dataset...")
    X, y = classifier.generate_synthetic_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the classifier
    print("\nTraining Naive Bayes classifier...")
    classifier.fit(X_train, y_train)
    
    # Evaluate the classifier
    print("\nEvaluating classifier performance...")
    classifier.evaluate(X_test, y_test)
    
    # Example prediction
    print("\nExample predictions:")
    example_docs = [
        "computer software programming code system",
        "market finance investment profit",
        "football player team championship game"
    ]
    predictions = classifier.predict(example_docs)
    for doc, pred in zip(example_docs, predictions):
        print(f"Document: '{doc}'\nPredicted class: {pred}\n")

if __name__ == "__main__":
    main()
