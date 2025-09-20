import pandas as pd
import numpy as np
import re
import sklearn.metrics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


class EconomicClassifier:
    def __init__(self, model_path:Path, initialize=False):
        # Construct paths to the model and vectorizer
        self.model_path = model_path / 'is_economy_model.joblib'
        self.vectorizer_path = model_path / 'is_economy_vectorizer.joblib'

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
        self.negative_set = {'outlook', 'arts & entertainment', 'style', 'movies', 'arts', 'life'}


    def vectorize_text(self, text_data):
        """Transform text data to vectorized format."""
        if text_data is None:
            return None
        return self.vectorizer.transform([text_data])


    def preprocess_text(self, text_data):
        """Lowercase, remove punctuation, numbers, and multiple spaces."""
        try:
            text_data = text_data.lower()
            text_data = re.sub(r'[^\w\s]', ' ', text_data)
            text_data = re.sub(r'\s+', ' ', text_data).strip()
            return text_data
        except Exception as e:
            logger.error(f"Preprocess error: {e}")
            return None


    def train_classifier(self, df:pd.DataFrame, data_column:str='paragrph_text', label_column:str='section', min_text_length: int=40):
        """Trains the logistic regression model using the DataFrame provided. and returns the test set."""
        df[label_column] = df[label_column].apply(self.preprocess_text)
        df['Label'] = df[label_column].apply(lambda x: 1 if x in self.positive_set else 0 if x in self.negative_set else None)
        df.dropna(subset=['Label'], inplace=True)
        df.dropna(subset=[data_column], inplace=True)
        logger.info(f"Training data size: {round(df.shape[0]*0.8)} test size is {round(df.shape[0]*0.2)}")
        df['Processed_Text'] = df[data_column].apply(self.preprocess_text)
        df = df[df['Processed_Text'].str.len() >= min_text_length]
        X = self.vectorizer.fit_transform(df['Processed_Text'])
        y = df['Label'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return X_test, y_test


    def is_economic_prob(self, text: str) -> float:
        """return probability of a text being economic."""
        processed_article = self.preprocess_text(text)
        vectorized_article = self.vectorize_text(processed_article)
        probability = self.model.predict_proba(vectorized_article)[:, 1][0]
        return round(probability, 4)


    def save_model(self):
        """Save the trained model and vectorizer to disk."""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)


    def evaluate_model(self, X_test, y_test, threshold: float):
        """Evaluates the model's performance on the test set"""
        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        return(sklearn.metrics.classification_report(y_test, predictions))
    
    
if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    SRC_PATH = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/src/')
    import sys
    sys.path.append(str(SRC_PATH))
    from config import *
    is_economic_model = is_economic_module.EconomicClassifier(model_path=IS_ECONOMIC_MODEL, initialize=True)  # Initialize the economic model
    tdm_parser = tdm_parser_module.TdmXmlParser()
    corpuses_dir = CORPUSES_PATH  #'all_dataset_file_names.txt' LosAngelesTimesDavid all_dataset_file_names.txt # Path to the input
    output_path = FILE_NAMES_PATH / 'is_economic_train_files.txt'  # Path to the output text file
    output_csv_path = PROJECT_DATA_PATH / 'train_data' / 'economic_classifier_train_data.csv'
    file_names_path = FILE_NAMES_PATH / 'is_economic_train_files.txt'  # 'all_dataset_file_names.txt'  # Path to the file containing names

    # crate df
    df = file_process.load_df_from_xml(corpuses_dir, file_names_path)
    
    a = is_economic_model.evaluate_on_df(df)
    a
    # Assuming df is a DataFrame with the necessary columns
    #X_test, y_test = is_economic_model.train_classifier(df, data_column='paragrph_text', lable_column='section')
    # classifier.save_model()
    #print(is_economic_model.evaluate_model(X_test, y_test, thrashold=0.7))
    #pass