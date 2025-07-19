import pandas as pd
import numpy as np
import re
import sklearn.metrics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path
import config 
class EconomicClassifier:
    def __init__(self, initialize=False):
        # Get the directory where the current module file is located
        self.module_dir = config.IS_ECONOMIC_MODEL
        
        # Construct paths to the model and vectorizer
        self.model_path = self.module_dir / 'is_economy_model.joblib'
        self.vectorizer_path = self.module_dir / 'is_economy_vectorizer.joblib'

        if initialize:
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, norm='l2')
            self.model = LogisticRegression(random_state=42)
            
        else:
            try:
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
            except Exception as e:
                
                print(f'Failed to load models: {e}')
                raise IOError("Model files could not be loaded.")
        self.positive_set = {'money', 'business', 'finance/business', 'business; part b; business desk', 'financial'}
        self.negative_set = {'outlook', 'arts & entertainment', 'style', 'movies', 'arts'}


    def vectorize_text(self, text_data):
        """Transform text data to vectorized format."""
        return self.vectorizer.transform([text_data])


    def preprocess_text(self, text_data):
        """Lowercase, remove punctuation, numbers, and multiple spaces."""
        try:
            text_data = text_data.lower()
            text_data = re.sub(r'[^\w\s]', ' ', text_data)
            text_data = re.sub(r'\s+', ' ', text_data).strip()
            return text_data
        except Exception as e:
            print(f'Preprocess error: {e}')
            return None

    def train_classifier(self, df, min_text_length=40):
        """Trains the logistic regression model using the DataFrame provided."""
        df['Label'] = df['Section'].apply(lambda x: 1 if x in self.positive_set else 0 if x in self.negative_set else None)
        df.dropna(subset=['Label'], inplace=True)
        df['Processed_Text'] = df['Text'].apply(self.preprocess_text)
        df = df[df['Processed_Text'].str.len() >= min_text_length]
        X = self.vectorizer.fit_transform(df['Processed_Text'])
        y = df['Label'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return X_test, y_test

    def is_economic_prob(self, text):
        """return probability of a text being economic."""
        processed_article = self.preprocess_text(text)
        vectorized_article = self.vectorize_text(processed_article)
        probability = self.model.predict_proba(vectorized_article)[:, 1][0]
        return probability

    def save_model(self):
        """Save the trained model and vectorizer to disk."""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)


    def evaluate_model(self, X_test, y_test):
        """Evaluates the model's performance on the test set"""
        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (probabilities > 0.7).astype(int)
        return(sklearn.metrics.classification_report(y_test, predictions))
    
    
