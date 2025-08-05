import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path


class TfidfKeywordExtractor:
    def __init__(self, model_path:Path, stop_words='english'):
        """
        Initialize the TfidfKeywordExtractor. If model_path is provided and the file exists,
        load the model automatically.
        Parameters:
            stop_words (str or list): Stop words to use in the TfidfVectorizer.
            model_path (str, optional): Path to a pre-trained TfidfVectorizer model.
        """
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.trained = False
        # Try to load the model automatically if model_path is provided and exists
        if model_path is not None:
            try:
                self.load(model_path)
            except Exception as e:
                print(f"Failed to load model from '{model_path}'. Error: {e}")
                

    def train(self, corpus):
        """
        Train the TfidfVectorizer on a provided corpus.
        Parameters:
            corpus (list of str): List of documents to train on.
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.trained = True

    def save(self, filepath):
        """
        Save the trained TfidfVectorizer to disk.
        
        Parameters:
            filepath (str): File path to save the model.
        """
        joblib.dump(self.vectorizer, filepath)
        print(f"TF-IDF Vectorizer model saved as '{filepath}'.")

    def load(self, filepath):
        """
        Load a TfidfVectorizer model from disk.
        Parameters:
            filepath (str): File path from which to load the model.
        """
        self.vectorizer = joblib.load(filepath)
        self.trained = True
        print(f"TF-IDF Vectorizer model loaded from '{filepath}'.")


    def transform(self, documents):
        """
        Transform new documents into TF-IDF vectors using the trained model.
        
        Parameters:
            documents (list of str): New documents to transform.
        
        Returns:
            Sparse matrix of TF-IDF vectors.
        """
        if not self.trained:
            raise ValueError("The vectorizer has not been trained or loaded yet.")
        return self.vectorizer.transform(documents)

    def extract_top_keywords(self, txt_str, top_n=None, score=True):
        """Extract the top_n keywords from a text."""
        # 1. Vectorize & flatten
        tfidf_vector    = self.transform([txt_str])
        dense_vector    = tfidf_vector.toarray().flatten()

        # 2. Only consider non-zero entries
        nonzero_idxs    = np.where(dense_vector > 0)[0]
        # 3. Sort those by score descending
        sorted_nonzero  = nonzero_idxs[np.argsort(dense_vector[nonzero_idxs])[::-1]]

        # 4. Limit to top_n if requested
        if top_n is not None:
            sorted_nonzero = sorted_nonzero[:top_n]

        # 5. Map back to feature names
        feature_names   = self.vectorizer.get_feature_names_out()
        if score:
            return [(feature_names[i], round(dense_vector[i], 4))
                    for i in sorted_nonzero]
        else:
            return [feature_names[i] for i in sorted_nonzero]



    def extract_keywords_from_documents(self, documents, top_n=None):
        """
        Extract top keywords for each document in a list of documents.
        Parameters:
            documents (list of str): List of new documents.
            top_n (int): Number of top keywords to extract from each document.
        Returns:
            List of lists, where each sublist contains tuples (keyword, score) for a document.
        """
        tfidf_matrix = self.transform(documents)
        keywords_list = []
        for i in range(tfidf_matrix.shape[0]):
            keywords = self.extract_top_keywords(tfidf_matrix[i], top_n)
            keywords_list.append(keywords)
        return keywords_list
    
    
    
if __name__ == '__main__':
    #parser = TdmXmlParser()
    #path = '/home/ec2-user/SageMaker/data/LosAngelesTimesDavid/422225372.xml'
    #val = parser.get_tag_value(path=path, tag_name='is_economic')
    #soup = parser.get_xml_soup(path)
    #print(val)
    tf_idf_extractor = TfidfKeywordExtractor(model_path=Path('/home/ec2-user/SageMaker/david/tdm-sentiment/src/topic_modeling/tf_idf_model/tfidf_vectorizer.pkl'))
    txt_str = "The quick brown fox jumps over the lazy dog. The dog was not happy about it."
    top_keywords = tf_idf_extractor.extract_top_keywords(txt_str)
    pass

